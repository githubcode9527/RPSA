import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from collections import deque
from torch import optim
from torch.autograd import Variable
from tqdm import tqdm

from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter
from matcher import EmbedMatcher


class Trainer(object):

    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)
            logging.info('{} : {}'.format(k, v))
        self.meta = not self.no_meta

        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            # test只加载id
            self.load_symbol2id()
            use_pretrain = False
        else:
            # train只加载预训练的嵌入
            self.load_embed()
        self.use_pretrain = use_pretrain

        # self.num_symbols = len(self.symbol2id.keys()) - 1 # one for 'PAD'
        # self.pad_id = self.num_symbols
        self.matcher = EmbedMatcher(self.embed_dim, num_rels=self.num_rels, num_ents=self.num_ents, rel_embed=self.rel_embed, ent_embed=self.ent_embed, dropout=self.dropout, batch_size=self.batch_size,use_pretrain=self.use_pretrain, finetune=self.fine_tune, dataset = self.dataset,beta=self.beta)
        self.matcher.cuda()

        self.epoch_num = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)    #可视化存放

        self.parameters = filter(lambda p: p.requires_grad, self.matcher.parameters())
        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)

        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[50000,100000], gamma=0.5)

        #self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        #self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)
        '''
        degree0 = open('degree0','w')
        for key,value in degrees.items():
            if value == 0 :
                degree0.write(key+'----'+str(0)+'\n')

        '''
        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self):
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.num_rels = len(self.rel2id.keys())
        self.rel2id['PAD'] = self.num_rels
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())
        self.ent2id['PAD'] = self.num_ents
        self.rel_embed = None
        self.ent_embed = None
        '''
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/relation2ids'))
        ent2id = json.load(open(self.dataset + '/ent2ids'))
        # 修改1
        i = 0
        for key in rel2id.keys():
            if key not in ['','OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None
        '''

    def load_embed(self):
        self.rel2id = json.load(open(self.dataset + '/relation2ids'))
        self.num_rels = len(self.rel2id.keys())
        self.ent2id = json.load(open(self.dataset + '/ent2ids'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx' or self.embed_model == 'TransE':
                # normalize the complex embeddings 预训练嵌入归一化
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == self.num_ents
            assert rel_embed.shape[0] == self.num_rels

            # 需要简化代码
            self.rel2id['PAD'] = self.num_rels
            self.ent2id['PAD'] = self.num_ents
            rel_embed_shape1 = rel_embed.shape[1]
            ent_embed_shape1 = ent_embed.shape[1]
            rel_embed = rel_embed.tolist()
            ent_embed = ent_embed.tolist()
            rel_embed.append(list(np.zeros((rel_embed_shape1,))))
            ent_embed.append(list(np.zeros((ent_embed_shape1,))))
            rel_embed = np.array(rel_embed)
            ent_embed = np.array(ent_embed)

            self.rel_embed = rel_embed
            self.ent_embed = ent_embed
            '''
            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['','OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key],:]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key],:]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())
            # 
            self.symbol2id = symbol_id
            self.symbol2vec = embeddings
            '''
    def build_connection(self, max_=100):

        self.connections = (np.ones((self.num_ents, max_, 2))).astype(int)
        self.connections[:,:,0] = self.num_rels
        self.connections[:,:,1] = self.num_ents
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.rel2id[rel], self.ent2id[e2]))   # 实体e1的邻居关系id和邻居id（元祖）
                self.e1_rele2[e2].append((self.rel2id[rel+'_inv'], self.ent2id[e1]))

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            # degrees.append(len(neighbors)) 
            degrees[ent] = len(neighbors)  # 超过最大邻居数的默认最大邻居数为度数
            self.e1_degrees[id_] = len(neighbors)    # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]  # connections是一个三维的矩阵：[头实体id,第几个邻居,0=关系]，[头实体id,第几个邻居,1=尾实体]  可以unsqueezed(-1)去掉第几点邻居的序号
        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.matcher.state_dict(), path)

    def load(self, path=None):
        if path == None :
            self.matcher.load_state_dict(torch.load(self.save_path))
        else :
            self.matcher.load_state_dict(torch.load(path))

    def get_meta(self, left, right):
        left_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in left], axis=0))).cuda()         #3维矩阵
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(torch.LongTensor(np.stack([self.connections[_,:,:] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')
        best_MRR = 0.0
        best_epoch = 0
        bad_counts = 0

        losses = deque([], self.log_every)

        for data in train_generate(self.dataset, self.batch_size, self.few, self.ent2id, self.e1rel_e2):
            # 解释：support是支持集[头实体id，尾实体id]        support_left/right是关于某一关系r支持集的头实体id集合和尾实体id集合
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data
            # TODO more elegant solution
            # get_meta获取头实体和尾实体的邻居
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            query_scores = self.matcher(False,query, support, query_meta, support_meta)
            false_scores = self.matcher(False,false, support, false_meta, support_meta)

            margin_ = query_scores - false_scores
            loss = F.relu(self.margin - margin_).mean()       

            losses.append(loss.item())
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.epoch_num % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                print('Epoch: {0}, Avg_batch_loss: {1:.6f}, lr:{2:.6f} '.format(self.epoch_num,np.mean(losses),lr))

                # self.save()
                # logging.info('AVG. BATCH_LOSS: {.2f} AT STEP {}'.format(np.mean(losses), self.batch_nums))
                self.writer.add_scalar('Avg_batch_loss', np.mean(losses), self.epoch_num)    # 训练的平均损失

            if (self.epoch_num % self.eval_every == 0 and self.epoch_num != 0) or self.epoch_num == self.max_epoch:

                logging.info('第 {} 次检验'.format(self.epoch_num//self.eval_every))
                logging.critical('第 {0} 次检验的学习率为: {1:.6f}'.format(self.epoch_num//self.eval_every, lr))

                hits10, hits5, hits1, mrr = self.eval(meta=self.meta)
                self.eval(mode='test',meta=self.meta)
                if mrr > best_MRR:
                    best_MRR = mrr
                    best_epoch = self.epoch_num
                    bad_counts = 0
                    logging.critical('Best model | MRR of valid set is {:.3f}'.format(best_MRR))
                    self.save(self.save_path + '_best')
                else :
                    logging.critical('Best MRR of valid set is {0:.3f} at {1} | bad count is {2}'.format(best_MRR, best_epoch, bad_counts))
                    bad_counts += 1

                self.writer.add_scalar('HITS10', hits10, self.epoch_num)
                self.writer.add_scalar('HITS5', hits5, self.epoch_num)
                self.writer.add_scalar('HITS1', hits1, self.epoch_num)
                self.writer.add_scalar('MRR', mrr, self.epoch_num)
            
            self.epoch_num += 1
            #self.scheduler.step()
            if self.epoch_num == self.max_epoch:
                logging.info('Train Finished!')
                logging.info('Final epochd result')
                self.eval(meta=self.meta, mode='test')
                self.save()
                self.load(self.save_path + '_best')
                self.eval(meta=self.meta, mode='test')
                break
            '''
            if bad_counts >= self.early_stopping_patience:
                logging.info('Early stopping!')
                logging.critical('Early stopping at epoch %d'.format(self.epoch_num))
                self.load(self.save_path + '_best')
                self.eval(meta=self.meta, mode='test')
                break
            '''


    def eval(self, mode='dev', meta=False):
        self.matcher.eval()
        
        ent2id = self.ent2id
        #symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

        rel2candidates = self.rel2candidates

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []

        for query_ in test_tasks.keys():

            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []
            test_task = test_tasks[query_]
            candidates = rel2candidates[query_]
            support_triples = test_task[:few]
            support_pairs = [[ent2id[triple[0]], ent2id[triple[2]]] for triple in support_triples]

            support_left = [self.ent2id[triple[0]] for triple in support_triples]
            support_right = [self.ent2id[triple[2]] for triple in support_triples]
            support_meta = self.get_meta(support_left, support_right)

            support = Variable(torch.LongTensor(support_pairs)).cuda()

            for triple in test_task[few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([ent2id[triple[0]], ent2id[triple[2]]])


                query_left = []
                query_right = []
                query_left.append(self.ent2id[triple[0]])
                query_right.append(self.ent2id[triple[2]])

                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0]+triple[1]]) and ent != true:
                        query_pairs.append([ent2id[triple[0]], ent2id[ent]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).cuda()


                query_meta = self.get_meta(query_left, query_right)
                scores = self.matcher(True, query, support, query_meta, support_meta)
                scores.detach()
                scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0/rank)
                mrr_.append(1.0/rank)


            logging.critical('{} Hits10:{:.3f}, Hits5:{:.3f}, Hits1:{:.3f} MRR:{:.3f}'.format(query_, np.mean(hits10_), np.mean(hits5_), np.mean(hits1_), np.mean(mrr_)))
            logging.info('Number of candidates: {}, number of text examples {}'.format(len(candidates), len(hits10_)))
            # print query_ + ':'
            # print 'HITS10: ', np.mean(hits10_)
            # print 'HITS5: ', np.mean(hits5_)
            # print 'HITS1: ', np.mean(hits1_)
            # print 'MRR: ', np.mean(mrr_)

        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))

        self.matcher.train()

        return np.mean(hits10), np.mean(hits5),np.mean(hits1), np.mean(mrr)

    def test_(self):
        self.load(self.save_path)
        logging.info('Pre-trained model loaded')
        
        self.eval(mode='test', meta=self.meta)
def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    args = read_options()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # setup random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    trainer = Trainer(args)
    if args.test:
        trainer.test_()
    else:
        trainer.train()
