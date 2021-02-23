import torch
import torch.nn as nn


def Limitloss(args, phs, prs, pts, nhs, nrs, nts):
    pos_distance = phs + prs - pts
    neg_distance = nhs + nrs - nts

    pos_score = torch.pow(torch.norm(pos_distance, p=2, dim=1), 2)  # square of L2 norm distance
    neg_score = torch.pow(torch.norm(neg_distance, p=2, dim=1), 2)  # square of L2 norm distance

    pos_loss = torch.sum(torch.relu(pos_score - args.pos_margin))
    neg_loss = torch.sum(torch.relu(args.neg_margin - neg_score))

    loss = args.pos_weight * pos_loss + args.neg_weight * neg_loss
    return loss


class Regularization(nn.Module):
    def __init__(self, model1, model2, weight_decay, p=2):
        """
        @param model:
        @param weight_decay:
        @param p:
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <= 0")
            exit(0)
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model1)
        self.weight_list.extend(self.get_weight(model2))

    def forward(self, model1, model2):
        self.weight_list = self.get_weight(model1)  # get the latest weights
        self.weight_list.extend(self.get_weight(model2))
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'embed' not in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0.0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss
