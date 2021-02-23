# coding=utf-8
import numpy as np


def read_ents(file):
    """
    read entity label-name file to get list of entities
    @param file:  entity label-name file
    @return:  list of entity labels (str) in the KG
    """
    lines = np.genfromtxt(file, dtype=np.dtype(str))
    ent_list = lines[:, 0].tolist()
    ent_list_int = []
    for item in ent_list:
        ent_list_int.append(int(item))
    return ent_list_int

def read_refs(file):
    """
    read reference entity pairs
    @param file:
    @return:
    """
    refs = np.genfromtxt(file, dtype=np.dtype(str))
    refs = refs.tolist()
    ref_list_int = []
    for item in refs:
        ref_list_int.append(tuple([int(item[0]), int(item[1])]))
    return ref_list_int

def read_triples(file):
    """
    read triples file, and return a list of triples and relations
    @param file:
    @return:
    """
    triple_list = []
    relation_set = set()
    triples = np.genfromtxt(file, dtype=np.dtype(str))
    for triple in triples:
        triple_list.append(tuple([int(triple[0]),int(triple[1]),int(triple[2]) ]))
        relation_set.add(int(triple[1]))
    return triple_list, relation_set

def process(folder):
    """
    @param folder:
    @return:
    """
    left_ents = read_ents(folder + 'ent_ids_1')
    # print(left_ents)
    print(len(left_ents))

    right_ents = read_ents(folder + 'ent_ids_2')
    # print(right_ents)
    print(len(right_ents))

    left_triples, left_rels = read_triples(folder + 'triples_1')
    right_triples, right_rels = read_triples(folder + 'triples_2')
    test = read_refs(folder + 'test')

    ent_list = list(set(left_ents) | set(right_ents))
    print(len(ent_list))
    rel_list = list(left_rels | right_rels)

    ent_num = len(ent_list)
    rel_num = len(rel_list)
    triple_list = left_triples + right_triples

    return triple_list, left_triples, right_triples, left_ents, right_ents, test, ent_num, rel_num
