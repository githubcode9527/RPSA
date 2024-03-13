import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from torch.nn import init


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # CoDEx-M-ver4 complex relation
    test_task  = json.load((open("CoDEx-M-ver4/test_tasks.json")))
    test_hr2t = defaultdict(list)
    for rel,triples in test_task.items():
        for triple in triples:
            test_hr2t[triple[0]+triple[1]].append()
'''
    test_task  = json.load((open("NELL/test_tasks.json")))
    test_n_n = defaultdict(list)
    for key,values in test_task.items():
        if key != 'concept:sportsgamesport':
            test_n_n[key] = values
    json.dump(test_n_n,open('NELL/test_n-n.json','w'))

    dataset = 'Wiki'
    triple2num = open('')
    traning = json.load((open("NELL/train_tasks.json")))
    print("----------------------训练集长度---------------------",len(traning))
    for key, value in traning.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    dev = json.load((open("NELL/dev_tasks.json")))
    print("----------------------验证集长度---------------------",len(dev))
    for key, value in dev.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    testing = json.load((open("NELL/test_tasks.json")))
    print("----------------------测试集长度---------------------",len(testing))
    for key, value in traning.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    rel2candidates_all = json.load(open("NELL/rel2candidates_all.json"))
    print("---------------rel2candidates_all大小------------------",len(rel2candidates_all))
    

 
    dataset = 'NELL'
    candidate_ori = json.load(open(dataset+'/rel2candidates_ori.json'))
    candidate = json.load(open(dataset+'/rel2candidates.json'))    
    candidate_ori_record = open(dataset+'/candidate_ori_record','w')
    candidate_record = open(dataset+'/candidate_record','w')
    for key,values in candidate_ori.items():
        candidate_ori_record.write(key+'------'+str(len(values))+'\n')
    for key,values in candidate.items():
        candidate_record.write(key+'--------'+str(len(values))+'\n')

    training = json.load(open("NELL/train_tasks.json"))
    tasks = open('NELL-rels.txt', 'w')
    tasks.write('-----------------------------------训练集---------------------------------------------------\n')
    for key, value in training.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------\n")

   set_LSTM_encoder = nn.LSTM(2 * 100, 2 * 100, 1, bidirectional=True)   # nn.LSTM(输入维度，隐藏层维度，循环神经网络层层数)

    input = torch.randn(5,1,200)
    support_g_encoder, support_g_state = set_LSTM_encoder(input)
    support_g_encoder = support_g_encoder[:,:,:200] + support_g_encoder[:,:,200:]
    
    print(support_g_encoder.size())


    pe = nn.Linear(10,10)
    init.xavier_normal_(pe.weight) 
    print(pe.bias)

    
     train_candidate = json.load(open('backupDataset/train_tasks_in_train.json'))
    tasks = open('backupDataset/NELL_train_task.txt', 'w')
    tasks.write('总数：'+str(len(train_candidate)))
    tasks.write("-------------------------------------------------------------------------------------------\n")
    for key, value in train_candidate.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------\n")

    training = json.load(open("Wiki/train_tasks.json"))
    new_training = json.load(open('Wiki/train_tasks_new.json'))
    tasks = open('new_tasks/wiki-rels.txt', 'w')
    tasks.write('-----------------------------------训练集---------------------------------------------------\n')
    for key, value in training.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------\n")

    tasks.write('-----------------------------------新的训练集---------------------------------------------------\n')
    for key, value in new_training.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------\n")
  
    traning = json.load((open("NELL/train_tasks.json")))
    dev = json.load(open('NELL/dev_tasks.json'))
    testig = json.load(open('NELL/test_tasks.json'))

    tasks = open('new_tasks/rels.txt','w')
    tasks.write('-----------------------------------训练集---------------------------------------------------')
    for key, value in traning.items():
            str0 = key + "-----------" + str(len(value)) + '\n'
            tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------")

    tasks.write('-----------------------------------验证集---------------------------------------------------')
    for key, value in dev.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------")

    tasks.write('-----------------------------------测试集---------------------------------------------------')
    for key, value in testig.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        tasks.write(str0)
    tasks.write("-------------------------------------------------------------------------------------------")
    tasks.close()
  

    known_rel = json.load(open('NELL/known_rels.json'))
    switch_test = open('backupDataset/switch_test.txt','w+')
    i = 0
    
            i += 1
    st = '总数-----------' + str(i)
    switch_test.write(st)
    switch_test.close()


    known_rel_info = open("backupDataset/known_rel_info_in_range.txt", 'w+')
    know_rels = json.load(open("NELL/known_rels.json"))
    print("---------------已知关系下的实体对大小------------------", len(know_rels))
    for key, value in know_rels.items():
        if len(value)>= 50 and len(value) <= 500:
            str0 = key + "-----------" + str(len(value)) + '\n'
            known_rel_info.write(str0)
    known_rel_info.close()

    rel2candidates_all_info = open("backupDataset/rel2candidates_all_info.txt",'w+')
    rel2candidates_all=json.load(open("NELL/rel2candidates_all.json"))
    print("--------------------rel2candidates_all关系个数--------------------------",len(rel2candidates_all))
    for key,value in rel2candidates_all.items():
        str0 = key + "-----------" + str(len(value)) + '\n'
        rel2candidates_all_info.write(str0)
 
    print("--------------实体对集合超过500-----------------------", i)
    
    support = torch.randn(10,200)
    query = torch.randn(100,1,200)
    wei = torch.matmul(query, support.transpose(0,1))
    wei = F.softmax(wei,dim=-1)
    print(wei.size())
    support_expand = support.repeat(100,1,1)
    print(support_expand.size())
    support_agg = wei.transpose(1,2) * support_expand
    print(support_agg.size())
    support_agg = torch.sum(support_agg,dim = 1)
    print(support_agg.size())

    i=0
    
    
   traning = json.load((open("NELL/train_tasks.json")))
    print("----------------------训练集长度---------------------",len(traning))
    for key, value in traning.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    dev = json.load((open("NELL/dev_tasks.json")))
    print("----------------------验证集长度---------------------",len(dev))
    for key, value in dev.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    testing = json.load((open("NELL/test_tasks.json")))
    print("----------------------测试集长度---------------------",len(testing))
    for key, value in traning.items():
        print(key)
        print(len(value))
    print("-----------------------------------------------")
    rel2candidates_all = json.load(open("NELL/rel2candidates_all.json"))
    print("---------------rel2candidates_all大小------------------",len(rel2candidates_all))

    rel2candidates = json.load(open("NELL/rel2candidates.json"))
    print("---------------rel2candidate大小-----------------------",len(rel2candidates))
    print(83//10)
    rel2candidates = json.load((open("NELL/rel2candidates.json")))
    print("----------------rel2candidates关系个数------------------",len(rel2candidates))
    for key,value in rel2candidates.items():
        print(key)
        print(len(value))

    

    p = torch.randn((2,3,4))
    q = torch.ones(2,3)*10
    print(p)
    print(q)
    q = q.unsqueeze(1).transpose(1,2)
    print(q)
    print(q.size())
    w = q * p
    print(w)
    print(torch.sum(w,dim=1))

    te = torch.randn((1,2,3,4))
    print(te)
    te1 = te.squeeze()
    print(te1)
    print(te1.size())
    we = torch.tensor(np.ones((2,3,4)) * 0.1)
    print(we)
    print(we.size())
    print(te1 * we)

    te = torch.BoolTensor(np.triu(np.ones((5, 5)), k=1))  # Upper triangular matrix
    print(te)
    t = torch.randn(5,5)
    print(t)
    print(F.softmax(t,dim=-1))
    t1 = t.masked_fill_(te,-1e9)
    print(t1)
    print(F.softmax(t1, dim=-1))
    te = torch.randn((1,10))
    print(te)
    te1 = TesM()
    #print(te1(te))


    te = torch.tensor(np.triu(np.ones((5,5)), k=1))  # Upper triangular matrix
    print(te)
    te[0][0] = te[1][0] = te[1][1] = -1e9
    print(te)
    te1 = F.softmax(te,dim=-1)
    print(te1)


     te = torch.tensor(np.ones((2, 3)))
    te1 = te.repeat(1, 1)
    print(te1)
    print(te1.size())
    te2 = te.unsqueeze(1)
    print(te2)
    print(te2.size())
    dec_input = torch.zeros(1, 0)
    print(dec_input)
    
    
    
    te = torch.randn(2,1,2,5)
    print(te)
    te1 = te.transpose(1,2)
    print(te1)
    print(te1.size())
    lis=[[1,2],[3,4],[5,6]];         #list转张量（矩阵）
    lis1=torch.tensor(lis)
    print(lis)
    print(lis1)
    print(lis1[:,0])
    lis2=np.stack([lis1[:,0] for i in range(5)], axis=1)
    lis4=torch.tensor(lis2)
    print(lis4)
    print(lis4.shape)
    lis3 = torch.randn(3, 5)
    print(lis3)

    t=torch.randn(5,4,2)
    t1=np.stack([t[0,:,:],t[1,:,:]],axis=0)
    print(t)

    t2=t[:,:,0].squeeze(-1)
    t3=t[:,:,1].squeeze(-1)
    print(t2)
    print(t3)
    t4=torch.cat((t2,t3),dim=-1)
    print(t4)
    
    te = torch.FloatTensor(np.stack([i for i in range(6)]))
    print(te)
    te1 = te.unsqueeze(1)
    print(te1)
    print(te1.shape())

    num_data = np.arange(1,13)     #np.arange(起点,终点,间隔) 间隔默认是1
    a = torch.ones(3, 4)
    b = num_data.reshape(3,4)     #修改数组维度
    print('a：',a)
    print('b：',b)
    print('a * b =',a*b)
  
    ls = [[1,1],[2,2]]
    ls0 = torch.FloatTensor(ls)
    print(ls0)
    ls1 = F.softmax(ls0,dim=1)
    print(ls1)



    batch_size = 5
    seq_len = 6
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=0)
    print(mask.unsqueeze(0))
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    print(mask)

    seq_k = seq_q = torch.randn(2,3)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    print(pad_attn_mask)
    print(pad_attn_mask.expand(batch_size, len_q, len_k))   # [batch_size, len_q, len_k]

    dl = torch.tensor([7,10,6,9,5,6])
    max_len = torch.max(dl)
    input_pos = [list(range(1,len + 1)) + [0] * (max_len - len) for len in dl]
    print(input_pos)
'''