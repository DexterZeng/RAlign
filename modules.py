import torch
import torch.nn as nn
import torch.nn.functional as f

from utils import trunc_norm_init, get_activation


class APP(nn.Module):
    def __init__(self, K: int, alpha: float):
        super(APP, self).__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, x, adj):
        h = x
        for k in range(self.K):
            x = torch.sparse.mm(adj, x)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        return x

class APPModel(nn.Module):
    def __init__(self, ent_num, dim, adj, appk):
        super(APPModel, self).__init__()

        self.ent_embed = nn.Parameter(torch.empty(ent_num, dim), requires_grad=True)
        self.gnn_layers = APP(K=int(appk), alpha=0.2)
        self.adj = adj
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.ent_embed.data)

    def forward(self):
        output = self.gnn_layers(self.ent_embed, self.adj)
        return output

class Model(nn.Module):
    def __init__(self, args, ent_num, rel_num, ):
        super(Model, self).__init__()
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.act = get_activation(args.activation)
        self.operator = args.operator
        self.batch_size = args.batch_size
        self.dim = args.dim

        self.rel_embed = nn.Parameter(torch.empty(rel_num, self.dim), requires_grad=True)
        # self.interact_embed = nn.Parameter(torch.empty(ent_num, self.dim), requires_grad=True)

        # self.interact = nn.Linear(self.dim, self.dim)
        # nn.init.orthogonal_(self.interact.weight.data)
        nn.init.xavier_normal_(self.rel_embed.data)

        if self.operator == 'projection':
            self._context_projection()
        elif self.operator == 'compression':
            self._context_compression()
        else:
            pass

    def forward(self, ent_embed, phs, prs, pts, nhs, nrs, nts):

        ent_embed.data = f.normalize(ent_embed.data)
        self.rel_embed.data = f.normalize(self.rel_embed.data)
        # interact_weight = self.act(self.interact(ent_embed.data))
        # self.interact_embed.data = f.normalize(interact_weight)

        ph_batch = ent_embed[phs]  # general entity embeddings for positive head entities
        pr_batch = self.rel_embed[prs]  # relation embeddings for positive relations
        pt_batch = ent_embed[pts]  # general entity embeddings for positive tail entities

        nh_batch = ent_embed[nhs]  # general entity embeddings for negative head entities
        nr_batch = self.rel_embed[nrs]  # relation embeddings for negative relations
        nt_batch = ent_embed[nts]  # general entity embeddings for negative tail entities

        phc_batch = ent_embed[phs]  # interaction entity embeddings for positive head entities
        ptc_batch = ent_embed[pts]  # interaction entity embeddings for positive tail entities

        nhc_batch = ent_embed[nhs]  # interaction entity embeddings for negative head entities
        ntc_batch = ent_embed[nts]  # interaction entity embeddings for negative tail entities

        if self.operator == 'compression':
            ##############################edge embedding##########################
            p_left = torch.cat((phc_batch, pr_batch), dim=1)
            p_right = torch.cat((pr_batch, ptc_batch), dim=1)
            p_left = f.normalize(p_left)
            p_right = f.normalize(p_right)

            p_left = self.act(self.context_compression['left_linear'](p_left))
            p_right = self.act(self.context_compression['right_linear'](p_right))
            p_left = f.normalize(p_left)
            p_right = f.normalize(p_right)

            p_edges = torch.cat((p_left, p_right), dim=1)
            p_edges = self.act(self.context_compression['linear'](p_edges))
            p_edges = f.normalize(p_edges)
            #################################################################################

            ##############################edge embedding##########################
            n_left = torch.cat((nhc_batch, nr_batch), dim=1)
            n_right = torch.cat((nr_batch, ntc_batch), dim=1)
            n_left = f.normalize(n_left)
            n_right = f.normalize(n_right)

            n_left = self.act(self.context_compression["left_linear"](n_left))
            n_right = self.act(self.context_compression['right_linear'](n_right))
            n_left = f.normalize(n_left)
            n_right = f.normalize(n_right)

            n_edges = torch.cat((n_left, n_right), dim=1)
            n_edges = self.act(self.context_compression['linear'](n_edges))
            n_edges = f.normalize(n_edges)
            #################################################################################
        elif self.operator == 'projection':
            ##############################edge embedding##########################
            pht_batch = torch.cat((phc_batch, ptc_batch), dim=1)
            pht_batch = f.normalize(pht_batch)

            pht_batch = self.act(self.context_projection['linear'](pht_batch))
            pht_batch = f.normalize(pht_batch)

            p_bias = torch.sum(pr_batch * pht_batch, dim=1, keepdim=True) * pht_batch
            p_edges = pr_batch - p_bias
            p_edges = f.normalize(p_edges)
            #################################################################################

            ##############################edge embedding##########################
            nht_batch = torch.cat((nhc_batch, ntc_batch), dim=1)
            nht_batch = f.normalize(nht_batch)

            nht_batch = self.act(self.context_projection['linear'](nht_batch))
            nht_batch = f.normalize(nht_batch)

            n_bias = torch.sum(nr_batch * nht_batch, dim=1, keepdim=True) * nht_batch
            n_edges = nr_batch - n_bias
            n_edges = f.normalize(n_edges)
            #################################################################################
        else:
            return ph_batch, pr_batch, pt_batch, nh_batch, nr_batch, nt_batch
        return ph_batch, p_edges, pt_batch, nh_batch, n_edges, nt_batch

    def _context_compression(self):
        self.context_compression = nn.ModuleDict({
            'left_linear': nn.Linear(2 * self.dim, self.dim),
            'right_linear': nn.Linear(2 * self.dim, self.dim),
            'linear': nn.Linear(2 * self.dim, self.dim)
        })
        left_linear_weight = self.context_compression['left_linear'].weight.data
        left_linear_bias = self.context_compression['left_linear'].bias.data

        right_linear_weight = self.context_compression['right_linear'].weight.data
        right_linear_bias = self.context_compression['right_linear'].bias.data

        linear_weight = self.context_compression['linear'].weight.data
        linear_bias = self.context_compression['linear'].bias.data

        self.context_compression['left_linear'].weight.data = trunc_norm_init(
            (left_linear_weight.size(0), left_linear_weight.size(1)))
        self.context_compression['left_linear'].bias.data = trunc_norm_init((left_linear_bias.size(0),))

        self.context_compression['right_linear'].weight.data = trunc_norm_init(
            (right_linear_weight.size(0), right_linear_weight.size(1)))
        self.context_compression['right_linear'].bias.data = trunc_norm_init((right_linear_bias.size(0),))

        self.context_compression['linear'].weight.data = trunc_norm_init(
            (linear_weight.size(0), linear_weight.size(1)))
        self.context_compression['linear'].bias.data = trunc_norm_init((linear_bias.size(0),))

    def _context_projection(self):
        self.context_projection = nn.ModuleDict({
            'linear': nn.Linear(2 * self.dim, self.dim)
        })

        linear_weight = self.context_projection['linear'].weight.data
        linear_bias = self.context_projection['linear'].bias.data

        self.context_projection['linear'].weight.data = trunc_norm_init(
            (linear_weight.size(0), linear_weight.size(1)))
        self.context_projection['linear'].bias.data = trunc_norm_init((linear_bias.size(0),))
