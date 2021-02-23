# coding=utf-8
import gc
import torch
import random

def cal_neighbors(ent_list, embeds, k):
    neighbor_dic = dict()
    sim_mat = torch.mm(embeds, embeds.t())

    _, sort_index = torch.topk(sim_mat, k + 1, largest=True, sorted=True)
    for i in range(sim_mat.size(0)):
        neighbor_dic[ent_list[i].item()] = ent_list[sort_index[i][1:]].tolist()

    del sim_mat
    gc.collect()
    return neighbor_dic


def trunc_sampling_multi(pos_triples, all_triples, dic, multi):
    """
    sample multiple negative triples for the
    specified (left or right) positive triples
    @param pos_triples:
    @param all_triples:
    @param dic:
    @param multi:
    @return:
    """
    neg_triples = list()
    for (h, r, t) in pos_triples:
        h_candidates = dic[h]
        h_candidates = random.sample(h_candidates, multi)
        negs = [(h_candidate, r, t) for h_candidate in h_candidates]
        neg_triples.extend(negs)

        t_candidates = dic[t]
        t_candidates = random.sample(t_candidates, multi)
        negs = [(h, r, t_candidate) for t_candidate in t_candidates]
        neg_triples.extend(negs)

    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_pos_batch(triple1_list, triple2_list, num1, num2, step):
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triple1_list):
        end1 = len(triple1_list)
    if end2 > len(triple2_list):
        end2 = len(triple2_list)
    pos_triples1 = triple1_list[start1: end1]
    pos_triples2 = triple2_list[start2: end2]
    return pos_triples1, pos_triples2


def generate_batch(left_triples_id, right_triples_id, num1, num2, step,
                   left_triples_id_set, right_triples_id_set, neighbors_dic1, neighbors_dic2, multi):
    """
    @param left_triple_id:
    @param right_triples_id:
    @param num1:
    @param num2:
    @param step:
    @param left_triples_id_set:
    @param right_triples_id_set:
    @param neighbors_dic1:
    @param neighbors_dic2:
    @param multi:
    @return:
    """
    # t = time.time()
    pos_triples1, pos_triples2 = generate_pos_batch(left_triples_id, right_triples_id, num1, num2, step)

    neg_triples = list()
    neg_triples.extend(trunc_sampling_multi(pos_triples1, left_triples_id_set, neighbors_dic1, multi))
    neg_triples.extend(trunc_sampling_multi(pos_triples2, right_triples_id_set, neighbors_dic2, multi))

    nhs_id, nrs_id, nts_id = list(), list(), list()
    for neg_h_id, neg_r_id, neg_t_id in neg_triples:
        nhs_id.append(neg_h_id)
        nrs_id.append(neg_r_id)
        nts_id.append(neg_t_id)

    phs_id, prs_id, pts_id = list(), list(), list()
    for pos_h_id, pos_r_id, pos_t_id in pos_triples1:
        phs_id.append(pos_h_id)
        prs_id.append(pos_r_id)
        pts_id.append(pos_t_id)
    for pos_h_id, pos_r_id, pos_t_id in pos_triples2:
        phs_id.append(pos_h_id)
        prs_id.append(pos_r_id)
        pts_id.append(pos_t_id)

    phs, prs, pts, nhs, nrs, nts = phs_id, prs_id, pts_id, nhs_id, nrs_id, nts_id
    # print('sample batch, time {:.3f} s'.format(time.time() - t))
    return phs, prs, pts, nhs, nrs, nts


def generate_batch_queue(que, left_triples_id, right_triples_id, num1, num2, steps,
                         left_triples_id_set, right_triples_id_set, neighbors_dic1, neighbors_dic2, multi):
    """
    @param que:
    @param left_triples_id:
    @param right_triples_id:
    @param num1:
    @param num2:
    @param steps:
    @param left_triples_id_set:
    @param right_triples_id_set:
    @param neighbors_dic1:
    @param neighbors_dic2:
    @param multi:
    @return:
    """
    for step in steps:
        phs, prs, pts, nhs, nrs, nts = generate_batch(left_triples_id, right_triples_id, num1, num2, step,
                                                      left_triples_id_set, right_triples_id_set,
                                                      neighbors_dic1, neighbors_dic2, multi)
        # print(phs)
        que.put((phs, prs, pts, nhs, nrs, nts))
