# coding=utf-8
import math

import torch
import torch.optim as optim
import torch.nn.functional as f

import numpy as np
from scipy.stats import truncnorm

from test_functions import get_hits, get_recip_hits, pref
from scipy.stats import rankdata

import time

def reciprocal(sim_mat, agg):
    sim_mat_r = sim_mat.T
    sim_mat = pref(sim_mat)
    # h1, h10, mrr = get_hits_ma(sim_mat)
    sim_mat_r = pref(sim_mat_r)
    ranks = rankdata(-sim_mat, axis=1)
    ranks_r = rankdata(-sim_mat_r, axis=1)

    if agg == 'arith':
        # ARITHMETIC MEAN
        rankfused = (ranks + ranks_r.T) / 2
        # h1, h10, mrr = get_hits_ma1(rankfused)
        # rankfused_r = (ranks_r + ranks.T) / 2
    else:
        # HARMONIC MEAN
        rankfused = 2 / (1 / ranks + 1 / ranks_r.T)
        # h1, h10, mrr = get_hits_ma1(rankfused_h)
        # rankfused_r_h = 2 / (1 / ranks_r + 1 / ranks.T)
    return rankfused

def BFS(graph, vertex):
    queue = []
    queue.append(vertex)
    looked = set()
    looked.add(vertex)
    while (len(queue) > 0):
        temp = queue.pop(0)
        nodes = graph[temp]
        for w in nodes:
            if w not in looked:
                queue.append(w)
                looked.add(w)
    return looked

def gen_adMtrx(x, y, aep_fuse):
    adMtrx = dict()
    for i in range(len(x)):
        x_ele = x[i]
        y_ele = y[i] + aep_fuse.shape[0]
        if x_ele not in adMtrx:
            ents = []
        else:
            ents = adMtrx[x_ele]
        ents.append(y_ele)
        adMtrx[x_ele] = ents
        if y_ele not in adMtrx:
            ents = []
        else:
            ents = adMtrx[y_ele]
        ents.append(x_ele)
        adMtrx[y_ele] = ents
    return adMtrx

def gen_adMtrx_more(x, y, rows, columns, aep_fuse):
    adMtrx1 = dict()
    for i in range(len(x)):
        x_ele = rows[x[i]]
        y_ele = columns[y[i]] + aep_fuse.shape[0]
        if x_ele not in adMtrx1:
            ents = []
        else:
            ents = adMtrx1[x_ele]
        ents.append(y_ele)

        adMtrx1[x_ele] = ents

        if y_ele not in adMtrx1:
            ents = []
        else:
            ents = adMtrx1[y_ele]
        ents.append(x_ele)
        adMtrx1[y_ele] = ents
    return adMtrx1

def get_blocks(adMtrx, allents):
    count1 = 0
    Graph = adMtrx
    leftents = allents
    blocks = []
    lenghs = []
    while len(leftents) > 0:
        vertex = list(leftents)[0]
        if vertex in Graph:
            matched = BFS(Graph, vertex)
        else:
            matched = {vertex}
        leftents = leftents.difference(matched)
        blocks.append(matched)
        lenghs.append(len(matched))
        if len(matched) == 1:
            count1 += 1
        # print()
    # print(blocks)
    print('Total blocks: ' + str(len(blocks)))
    # print(lenghs)
    print('Total blocks with length 1: ' + str(count1))
    # print(count1)
    # print(lenghs[0])
    return blocks

def ana_blocks(blocks, aep_fuse, maxtruth, correct_coun, recip_flag, agg):
    all1s = []
    refined_blocks = 0
    for block in blocks:
        if len(block) > 1:
            refined_blocks += 1
            rows = []
            columns = []

            for item in block:
                if item < aep_fuse.shape[0]:
                    rows.append(item)
                    if item + aep_fuse.shape[0] in block:
                        maxtruth += 1
                else:
                    columns.append(item - aep_fuse.shape[0])

            tempM = aep_fuse[rows][:, columns]

            if recip_flag is False:
                for i in range(tempM.shape[0]):
                    rank = (-tempM[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1
            else:
                rankfused = reciprocal(tempM, agg)
                for i in range(rankfused.shape[0]):
                    rank = (rankfused[i, :]).argsort()
                    if rows[i] == columns[rank[0]]:
                        correct_coun += 1
        else:
            all1s.extend(block)

    print('Total blocks after refinement: ' + str(refined_blocks+1))
    return all1s, maxtruth, correct_coun

def dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, flag, recip_flag, agg):
    rows = []
    columns = []

    for item in all1s:
        if item < aep_fuse.shape[0] and item + aep_fuse.shape[0] in all1s:
            maxtruth += 1
        if item < aep_fuse.shape[0]:
            rows.append(item)
        if item > aep_fuse.shape[0]:
            columns.append(item-aep_fuse.shape[0])

    tempM = aep_fuse[rows][:,columns]

    if flag is True:
        if recip_flag is False:
            for i in range(tempM.shape[0]):
                rank = (-tempM[i, :]).argsort()
                if rows[i] == columns[rank[0]]:
                    correct_coun += 1
        else:
            # # reciprocal
            rankfused = reciprocal(tempM, agg)
            for i in range(rankfused.shape[ 0]):
                rank = (rankfused[i, :]).argsort()
                if rows[i] == columns[rank[0]]:
                    correct_coun += 1

        print()
        print('Max truth: ' + str(maxtruth))
        print('Total correct: ' + str(correct_coun))

        print()
        print("Hits@1: " + str(correct_coun*1.0/aep_fuse.shape[0]))
    return rows, columns, tempM

def blocking(embed1, embed2, agg):
    print('***********************************************************')
    print('*********             Start blocking             **********')
    print('***********************************************************')

    recip_flag = True
    t = time.time()
    aep_fuse = torch.mm(embed1, embed2.t()).cpu().detach().numpy()  # should be the similarity score

    thres = 0.65
    x, y = np.where(aep_fuse > thres)
    adMtrx = gen_adMtrx(x, y, aep_fuse)

    allents = set()
    for i in range(aep_fuse.shape[0] + aep_fuse.shape[1]):
        allents.add(i)
    blocks = get_blocks(adMtrx, allents)
    del adMtrx
    del allents
    del x, y
    # evaluation!!!!
    maxtruth = 0
    correct_coun = 0

    all1s, maxtruth, correct_coun = ana_blocks(blocks, aep_fuse, maxtruth, correct_coun,recip_flag, agg)

    rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s, aep_fuse, False,
                                           recip_flag, agg)
    print('Lagrest block... (all 1s): ' + str(len(all1s)))
    del all1s

    print('\n*********************************************************')
    thres2 = 0.5
    x, y = np.where(tempM > thres2)
    adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)

    allents = []
    allents.extend(rows)
    for item in columns:
        allents.append(item + aep_fuse.shape[0])
    allents = set(allents)
    newblocks = get_blocks(adMtrx1, allents)
    del adMtrx1
    del allents

    all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag, agg)

    rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, False, recip_flag, agg)
    print('Lagrest block... (all 1s): ' + str(len(all1s_new)))
    del all1s_new


    print('\n**********************************************************')
    thres2 = 0.4
    x, y = np.where(tempM > thres2)
    adMtrx1 = gen_adMtrx_more(x, y, rows, columns, aep_fuse)

    allents = []
    allents.extend(rows)
    for item in columns:
        allents.append(item + aep_fuse.shape[0])
    allents = set(allents)

    newblocks = get_blocks(adMtrx1, allents)
    del adMtrx1
    del allents

    all1s_new, maxtruth, correct_coun = ana_blocks(newblocks, aep_fuse, maxtruth, correct_coun, recip_flag, agg)

    rows, columns, tempM = dirtect_process(maxtruth, correct_coun, all1s_new, aep_fuse, True,recip_flag, agg)

    del all1s_new

def trunc_norm_init(size):
    """
    @param size: the size of the tensor to be initialized
    @return: the initialized tensor
    """
    tensor = truncnorm.rvs(-2, 2, 0, 1, size=size)
    tensor = torch.from_numpy(tensor)
    return tensor.float()


def get_optimizer(name, parameters, lr, weight_decay=0.0):
    """
    initialize parameters initializer
    @param name: name for optimizer
    @param parameters: model parameters
    @param lr: learning rate
    @param weight_decay: weight_decay rate
    @return: the specified optimizer
    """
    if name == 'sgd':
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay)

    elif name == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)

    elif name == 'adagrad':
        return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)

    elif name == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

    elif name == 'adamax':
        return optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)

    else:
        raise Exception('Unsupported optimizer: {}'.format(name))


def get_activation(activation_string):
    """
    get the activation function accroding to the activation name
    @param activation_string:
    @return:
    """
    act = activation_string.lower()
    if act == 'tanh':
        return torch.tanh

    elif act == 'relu':
        return torch.relu

    elif act == 'selu':
        return torch.relu

    elif act == 'sigmoid':
        return torch.sigmoid

    elif act == 'gelu':
        return f.gelu

    elif act == 'leaky_relu':
        return f.leaky_relu

    else:
        raise ValueError("Unsupported activation function: {}".format(act))


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


def get_mat(ent_num, triples):
    degree = [1] * ent_num
    for triple in triples:
        if triple[0] != triple[2]:
            degree[triple[0]] += 1
            degree[triple[2]] += 1

    adj = {}
    for triple in triples:
        if triple[0] == triple[2]:
            continue

        if (triple[0], triple[2]) not in adj:
            adj[(triple[0], triple[2])] = 1
        else:
            pass

        if (triple[2], triple[0]) not in adj:
            adj[(triple[2], triple[0])] = 1
        else:
            pass
    for i in range(ent_num):
        adj[(i, i)] = 1

    return adj, degree


def get_sparse_adj(ent_num, triples):
    """
    get a sparse normalized adj tensor based on relational triples
    @param ent_num:  total number of entities
    @param triples:  relaitonal triples list
    @return:
    """
    adj, degree = get_mat(ent_num, triples)

    indices = []
    values = []

    for fir, sec in adj:
        indices.append((sec, fir))
        values.append(adj[(fir, sec)] / math.sqrt(degree[fir]) / math.sqrt(degree[sec]))

    indices = torch.tensor(indices).t()

    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=[ent_num, ent_num])

    return adj, degree
