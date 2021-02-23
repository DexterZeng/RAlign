# coding=utf-8
import gc
import time
import copy
import torch
import numpy as np
import multiprocessing
from scipy.stats import rankdata


def div_list(ls, n):
    ls_len = len(ls)
    if n <= 0 or 0 == ls_len:
        return [ls]

    if n > ls_len:
        return [ls]
    elif n == ls_len:
        return [[i] for i in ls]
    else:
        j = ls_len // n
        ls_return = []

        for i in range(0, (n - 1) * j, j):
            ls_return.append(ls[i:i + j])
        ls_return.append(ls[(n - 1) * j:])
        return ls_return

def pairwise_distance(x, y):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(dist, 0.0, np.inf)

def csls_values(sim_mat, k):
    nearest_k, _ = torch.topk(sim_mat, k, dim=1, largest=True, sorted=False)
    sim_values = torch.mean(nearest_k, dim=1)
    return sim_values

def csls_sim(embed1, embed2, k):
    sim_mat = torch.mm(embed1, embed2.t())
    if k <= 0:
        return sim_mat
    csls1 = csls_values(sim_mat, k)
    csls2 = csls_values(sim_mat.t(), k)

    csls_sim_mat = 2 * sim_mat.t() - csls1
    csls_sim_mat = csls_sim_mat.t() - csls2

    del sim_mat
    gc.collect()
    return csls_sim_mat

def pref(sim_mat):
    nearest_values2 = np.max(sim_mat.T, axis=1)
    csls_sim_mat = sim_mat - nearest_values2
    return csls_sim_mat

def get_hits_ma(sim, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = (-sim[i, :]).argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] / sim.shape[0], top_lr[1] / sim.shape[0], mrr_sum_l / sim.shape[0])
    print(msg)
    return top_lr[0] / sim.shape[0], top_lr[1] / sim.shape[0], mrr_sum_l / sim.shape[0]

def get_hits_ma1(sim, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        rank = (sim[i, :]).argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f\n' % (top_lr[0] /sim.shape[0], top_lr[1] / sim.shape[0], mrr_sum_l / sim.shape[0])
    print(msg)
    return top_lr[0] / sim.shape[0], top_lr[1] / sim.shape[0], mrr_sum_l / sim.shape[0]

gpu = torch.device('cuda:0')
cpu = torch.device('cpu')

def get_hits(embed1, embed2):
    """
    compute the specified measures on the test set
    @param embed1: embedding matrix for the left entities of thes test set
    @param embed2: embedding matrix for the right entities of the test set
    @param csls_k: csls k value used in csls similarity calculation
    @param device: gpu device
    @param topk: top k tuple
    @return:
    """
    sim_mat = torch.mm(embed1, embed2.t()).cpu().detach().numpy()

    # sim_mat = csls_sim(embed1, embed2, k=csls_k)  # unidirectional alignment mechanism
    # sim_mat = sim_mat.cpu().detach().numpy()
    precision, _, _ = get_hits_ma(sim_mat, top_k=(1,10))
    del sim_mat
    return precision, precision


def get_recip_hits(embed1, embed2, nameinfo, folder, agg):
    sim_mat = torch.mm(embed1, embed2.t()).cpu().detach().numpy()
    if nameinfo is True:
        str_sim = np.load(folder + '/string_mat_train.npy')
        str_sim = str_sim[:sim_mat.shape[0], :sim_mat.shape[0]]
        # get_hits_ma(str_sim, test)
        aep_n = np.load(folder + '/name_mat_train.npy')
        if 'fr_en' in folder or 'en_fr' in folder or 'en_de' in folder:
            weight_stru = 0.33
            weight_text = 0.33
            weight_string = 0.33
        else:
            weight_stru = 0.45
            weight_text = 0.45
            weight_string = 0.1
        sim_mat = (sim_mat * weight_stru + aep_n * weight_text + str_sim * weight_string)

    # print('Pure structural: ')
    # h1, h10, mrr = get_hits_ma(sim_mat)
    sim_mat_r = sim_mat.T

    # print('Preference matrix: ')
    sim_mat = pref(sim_mat)
    # h1, h10, mrr = get_hits_ma(sim_mat)
    sim_mat_r = pref(sim_mat_r)

    ranks = rankdata(-sim_mat, axis=1)
    ranks_r = rankdata(-sim_mat_r, axis=1)

    if agg == 'arith':
        print('Arithmetic mean: ')
        rankfused = (ranks + ranks_r.T) / 2
        h1, h10, mrr = get_hits_ma1(rankfused)
        rankfused_r = (ranks_r + ranks.T) / 2
    else:
        print('Harmonic mean: ')
        rankfused = 2 / (1 / ranks + 1 / ranks_r.T)
        h1, h10, mrr = get_hits_ma1(rankfused)
        rankfused_r = 2 / (1 / ranks_r + 1 / ranks.T)
    return rankfused