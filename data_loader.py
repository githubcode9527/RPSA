import json
import random
from tqdm import tqdm
import logging

def train_generate(dataset, batch_size, few, ent2id, e1rel_e2):
    logging.info('LOADING TRAINING DATA')
    train_tasks = json.load(open(dataset + '/train_tasks.json'))
    logging.info('LOADING CANDIDATES')
    rel2candidates = json.load(open(dataset + '/rel2candidates.json'))
    task_pool = list(train_tasks.keys())
    num_tasks = len(task_pool)
    rel_idx = 0

    while True:
        if rel_idx % num_tasks == 0:
            #打乱任务池
            random.shuffle(task_pool)
        query = task_pool[rel_idx % num_tasks]  #随机抽取任务（关系key）
        rel_idx += 1
        candidates = rel2candidates[query]     #选取候选实体

        if len(candidates) <= 20:
            # print 'not enough candidates'
            continue

        train_and_test = train_tasks[query]

        random.shuffle(train_and_test)

        support_triples = train_and_test[:few]

        support_pairs = [[ent2id[triple[0]], ent2id[triple[2]]] for triple in support_triples] #构造支持集的实体对[[头实体id，尾实体id]，....，[]]

        support_left = [ent2id[triple[0]] for triple in support_triples]      #头实体id列表：[实体id1,实体id2,...]
        support_right = [ent2id[triple[2]] for triple in support_triples]

        all_test_triples = train_and_test[few:]    #除去支持集的三元组，剩下的三元组用于训练集即查询集   这个查询集，需要随机打乱随机抽数据出来训练

        if len(all_test_triples) == 0:
            continue

        if len(all_test_triples) < batch_size:
            query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]   #随机抽取batch_size个查询三元组，补充到测试集满足batch_size
        else:
            query_triples = random.sample(all_test_triples, batch_size)

        query_pairs = [[ent2id[triple[0]], ent2id[triple[2]]] for triple in query_triples]

        query_left = [ent2id[triple[0]] for triple in query_triples]
        query_right = [ent2id[triple[2]] for triple in query_triples]

        false_pairs = []   #构造错误三元组
        false_left = []
        false_right = []
        for triple in query_triples:
            e_h = triple[0]
            rel = triple[1]
            e_t = triple[2]
            while True:
                #condidate包括真实的尾实体和假的尾实体
                noise = random.choice(candidates)     #随机抽取这个关系下的实体作为噪声尾实体（错误尾实体）
                if (noise not in e1rel_e2[e_h+rel]) and noise != e_t:
                    break
            false_pairs.append([ent2id[e_h], ent2id[noise]])
            false_left.append(ent2id[e_h])
            false_right.append(ent2id[noise])
        yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right







