# coding=utf-8
import os
import gc
import time
import datetime

from collections import defaultdict
import math
import multiprocessing

import random
import numpy as np

import torch
import torch.nn.functional as f
import argparse

from modules import APPModel, Model
from data_utils import process

from utils import get_optimizer, div_list, get_sparse_adj, blocking
from sample import generate_batch_queue, cal_neighbors
from loss import Regularization, Limitloss
from test_functions import get_hits, get_recip_hits

import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(args):
    borderline = 40000
    print(args)
    time_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print()
    print(time_str)
    print()
    gpu = torch.device('cuda:0')  # set gpu device
    cpu = torch.device('cpu')  # set cpu devcice

    triple_list, left_triples, right_triples, left_ents, right_ents, \
    test, ent_num, rel_num = process(args.folder)


    print('Loading completed... ')

    adj, degree = get_sparse_adj(ent_num, triple_list)
    adj = adj.to(gpu)

    left_ents = torch.tensor(left_ents).to(gpu)
    right_ents = torch.tensor(right_ents).to(gpu)

    n = len(test)
    # random.shuffle(test)
    testall = copy.deepcopy(test)
    testall = torch.tensor(testall)

    valid = test[:n // 7 * 1]
    test = test[n // 7 * 1:]

    valid = torch.tensor(valid)
    test = torch.tensor(test)

    if ent_num > borderline:
        num = 100000
        test = test.to(cpu)
        testall = testall.to(cpu)
    else:
        num = 15000
    truncated_num = int(num * (1 - args.truncated_epsilon))

    triple_num = len(triple_list)
    num_batch = math.ceil(triple_num / args.batch_size)  # number of batches
    stepss = div_list(list(range(num_batch)), args.threads)

    left_triples_set = set(left_triples)
    right_triples_set = set(right_triples)

    num1 = int(len(left_triples) / (len(left_triples) + len(right_triples)) * args.batch_size)
    num2 = args.batch_size - num1

    app = APPModel(ent_num=ent_num, dim=args.dim, adj=adj, appk = args.appk)
    model = Model(args=args, ent_num=ent_num, rel_num=rel_num)

    app.to(gpu)
    model.to(gpu)

    # optimizer
    optimizer = get_optimizer(args.optimizer,
                              [{'params': app.parameters(), 'lr': 0.005},
                               {'params': model.parameters()}],
                              lr=args.lr)

    # leanring rate scheduler
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # regularization loss
    Reg = Regularization(model1=app,
                         model2=model,
                         weight_decay=args.weight_decay)

    neighbors_dic1 = None
    neighbors_dic2 = None

    best_prec = 0.0  # best precision till now


    print("start training!!!! ")
    for epoch in range(1, args.epochs + 1):
        ################################################################################################################
        ####################################################Sampling####################################################
        app.train()
        model.train()
        optimizer.zero_grad()
        ent_embed = app()
        ent_embed = f.normalize(ent_embed)
        if epoch % args.freq == 1:  # sample negative triples every several epochs
            left_embed = ent_embed[left_ents]
            right_embed = ent_embed[right_ents]
            if ent_num > borderline:
                left_embed = left_embed.to(cpu)
                right_embed = right_embed.to(cpu)
            del neighbors_dic1
            del neighbors_dic2
            gc.collect()

            print('Start generating neighbors')
            t1 = time.time()  # time for starting calculating negative neighbors
            neighbors_dic1 = cal_neighbors(left_ents, left_embed, truncated_num)
            neighbors_dic2 = cal_neighbors(right_ents, right_embed, truncated_num)
            print()
            print('generate neighbors: {:.3f} s'.format(time.time() - t1))
        ################################################################################################################
        ####################################################Training####################################################
        t2 = time.time()  # time for starting training in each epoch
        random.shuffle(left_triples)
        random.shuffle(right_triples)
        batch_queue = multiprocessing.Queue(num_batch)
        for i in range(args.threads):
            multiprocessing.Process(target=generate_batch_queue,
                                    args=(batch_queue,
                                          left_triples,
                                          right_triples,
                                          num1,
                                          num2,
                                          stepss[i],
                                          left_triples_set,
                                          right_triples_set,
                                          neighbors_dic1,
                                          neighbors_dic2,
                                          args.neg_size)).start()
        epoch_loss = 0.0
        for step in range(num_batch):
            phs, prs, pts, nhs, nrs, nts = batch_queue.get()

            phs, prs, pts = torch.tensor(phs), torch.tensor(prs), torch.tensor(pts)
            nhs, nrs, nts = torch.tensor(nhs), torch.tensor(nrs), torch.tensor(nts)

            phs, prs, pts = phs.to(gpu), prs.to(gpu), pts.to(gpu)
            nhs, nrs, nts = nhs.to(gpu), nrs.to(gpu), nts.to(gpu)

            ph_batch, p_edges, pt_batch, nh_batch, n_edges, nt_batch = model(ent_embed, phs, prs, pts, nhs, nrs, nts)

            loss = Limitloss(args, ph_batch, p_edges, pt_batch, nh_batch, n_edges, nt_batch)
            reg_loss = Reg(app, model)

            loss += reg_loss

            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item()
        # scheduler.step(epoch)
        print()
        print('Training Epoch [{}/{}], Loss = {:.3f} | time = {:.3f} s'.
              format(epoch, args.epochs, epoch_loss / num_batch, time.time() - t2))
        #####################################################Validatinging##############################################
        ################################################################################################################
        with torch.no_grad():
            app.eval()
            model.eval()
            embed1 = ent_embed[valid[:, 0]] # validation set to check the performance...
            embed2 = ent_embed[valid[:, 1]]
            embed1 = f.normalize(embed1)
            embed2 = f.normalize(embed2)
            prec, hits = get_hits(embed1, embed2)
            if prec > best_prec:
                best_prec = prec
                print("save entity embeddings of epoch {}".format(epoch))
                torch.save(ent_embed, 'embed_{}.pt'.format(time_str))

    #####################################################Testing########################################################
    ####################################################################################################################
    print()

    print("Testing......\n")
    ent_embed = torch.load('embed_{}.pt'.format(time_str), map_location=gpu)
    ent_embed = f.normalize(ent_embed)

    embed1 = ent_embed[testall[:, 0]]
    embed2 = ent_embed[testall[:, 1]]
    if ent_num > borderline:
        embed1 = embed1.to(cpu)
        embed2 = embed2.to(cpu)
    #     _, hits = get_hits(embed1, embed2)
    # else:
    #     _, hits = get_hits(embed1, embed2)


    # whether blocking or not!!
    if args.pb:
        blocking(embed1, embed2, args.agg)
    else:
        get_recip_hits(embed1, embed2, False, args.folder, args.agg)

def make_print_to_file(fileName, path='./'):
    import os
    import sys

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))

if __name__ == "__main__":
    t = time.time()

    parser = argparse.ArgumentParser(description='Structure Reserving Network with Relation Contextualization')
    parser.add_argument('--operator', type=str, default='projection',
                        choices=('projection', 'compression', 'none'))
    parser.add_argument('--folder', type=str, default='../dbp15k/zh_en/0_3/')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--optimizer', type=str, default='adamax',
                        choices=('sgd', 'rmsprop', 'adagrad', 'adam', 'adamax'))
    parser.add_argument('--activation', type=str, default='relu',
                        choices=('tanh', 'relu', 'selu', 'sigmoid'))
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--pos_margin', type=float, default=0.2)
    parser.add_argument('--neg_margin', type=float, default=2)
    parser.add_argument('--neg_weight', type=float, default=1.2) #1.5 change from 1.0 to 0.8
    parser.add_argument('--pos_weight', type=float, default=1)

    parser.add_argument('--truncated_epsilon', type=float, default=0.95)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--csls_k', type=int, default=5)

    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dim', type=int, default=300)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--freq', type=int, default=10)  # sampling frequency change from 10 to 5
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--neg_size', type=int, default=3) #3 5(ja not enough)
    parser.add_argument('--highway', type=bool, default=True)

    parser.add_argument('--appk', type=str, default='2')
    parser.add_argument('--agg', type=str, default='arith')
    parser.add_argument('--pb', type=bool, default=True)

    args = parser.parse_args()
    make_print_to_file(args.folder.replace('.', '').replace('/', '_'), path='')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    run(args=args)

    print('Total time = {:.3f} s'.format(time.time() - t))
