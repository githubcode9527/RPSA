import logging

import torch
from torch import nn
from torch.nn import init
from modules import *
import torch.nn.functional as F


class EmbedMatcher(nn.Module):
    def __init__(self, embed_dim, num_rels, num_ents, rel_embed=None, ent_embed=None, dropout=0.2, batch_size=64, use_pretrain=True, finetune=True, dataset='NELL',beta=5.0):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.rel_emb = nn.Embedding(num_rels + 1, embed_dim, padding_idx=num_rels) 
        self.ent_emb = nn.Embedding(num_ents + 1, embed_dim, padding_idx=num_ents)
        self.dataset = dataset

        # 关系层参数配置
        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)            
        self.gcn_p = nn.Linear(self.embed_dim, 1)

        # 实体层参数配置
        self.gcn_w2 = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_q = nn.Linear(self.embed_dim, 1)

        # aotoencoder 自动编码器 配置参数
        self.set_LSTM_encoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional=False)   
        self.set_LSTM_decoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional=False)


        self.leak = nn.LeakyReLU(0.2)             
        self.dropout = nn.Dropout(dropout)


        init.xavier_normal_(self.gcn_w.weight)     
        init.xavier_normal_(self.gcn_p.weight)
        init.xavier_normal_(self.gcn_w2.weight)
        init.xavier_normal_(self.gcn_q.weight)


        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.rel_emb.weight.data.copy_(torch.from_numpy(rel_embed)) 
            self.ent_emb.weight.data.copy_(torch.from_numpy(ent_embed))
           
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.ent_emb.weight.requires_grad = False
                self.rel_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
    
    

    def neighbor_encoder(self, head_id, connections, num_neighbors):
        num_neighbors = num_neighbors.unsqueeze(1)           
        relations = connections[:, :, 0].squeeze(-1)          
        entities = connections[:, :, 1].squeeze(-1)           


        head_id = head_id.unsqueeze(0).transpose(0,1)
        head_id_stack = head_id.expand(head_id.size()[0], relations.size()[1])

        # 转换为嵌入
        head_embed = self.dropout(self.ent_emb(head_id_stack))    
        rel_embeds = self.dropout(self.rel_emb(relations))        
        ent_embeds = self.dropout(self.ent_emb(entities))          

        concat_embeds = torch.cat((head_embed, rel_embeds), dim=-1)  
        a_hr = self.gcn_w(concat_embeds)
        
        weight_hr = F.softmax(self.leak(self.gcn_p(a_hr).squeeze(-1)), dim=-1)

        # 拼接计算第二层注意力：实体层注意力->拼接a_hr和尾实体
        concat_embeds1 = torch.cat((a_hr, ent_embeds),dim=-1)
        b_hrt = self.gcn_w2(concat_embeds1)
        weight_hrt = F.softmax(self.leak(self.gcn_q(b_hrt).squeeze(-1)),dim=-1)

        # 第三层注意力：三元组注意力
        triple_weight = weight_hr * weight_hrt                                        
        triple_weight = triple_weight.unsqueeze(2)

        head_aggregate = torch.sum(triple_weight * b_hrt, dim=1)

        # 残差再次聚合
        head_aggregate_add = head_aggregate + self.ent_emb(head_id).squeeze(1)
        head_aggregate_mul = self.ent_emb(head_id).squeeze(1) * head_aggregate

        head_combination = (self.leak(head_aggregate_add) + self.leak(head_aggregate_mul))/2
      
        
        return head_combination

    def forward(self,iseval, query, support, query_meta=None, support_meta=None):

       
        batch = query.size()[0]
        few = support.size()[0]
        query_left_connections, query_left_degrees, query_right_connections, query_right_degrees = query_meta
        support_left_connections, support_left_degrees, support_right_connections, support_right_degrees = support_meta

        query_left = self.neighbor_encoder(query[:, 0], query_left_connections, query_left_degrees)
        query_right = self.neighbor_encoder(query[:, 1], query_right_connections, query_right_degrees)

        support_left = self.neighbor_encoder(support[:, 0], support_left_connections, support_left_degrees)
        support_right = self.neighbor_encoder(support[:, 1], support_right_connections, support_right_degrees)

        query_neighbor = torch.cat((query_left, query_right), dim=-1)  # tanh
        support_neighbor = torch.cat((support_left, support_right), dim=-1)  # tanh

        support_g = self.support_encoder(support_neighbor)
        query_g = self.support_encoder(query_neighbor)

        

        # LSTM autoencoder
        support_g_0 = support_g.view(few, 1, 2 * self.embed_dim)   
        support_g_encoder, support_g_state = self.set_LSTM_encoder(support_g_0)
        if not iseval:
            
            support_g_0.retain_grad()
            
            support_g_decoder = support_g_encoder[-1].view(1, -1, 2 * self.embed_dim)
            decoder_set = []
            support_g_decoder_state = support_g_state
            for idx in range(few):
                support_g_decoder, support_g_decoder_state = self.set_LSTM_decoder(support_g_decoder, support_g_decoder_state)
                decoder_set.append(support_g_decoder)

            decoder_set = torch.cat(decoder_set, dim=0)
            ae_loss = nn.MSELoss()(support_g_0, decoder_set.detach())
            self.zero_grad()
            ae_loss.backward(retain_graph=True)
            grad_meta = support_g_0.grad
            support_g_0 = support_g_0 - self.beta*grad_meta

        support_g_encoder = support_g_encoder.view(few, 2 * self.embed_dim)
        support_g_encoder = (support_g_0.view(few, 2 * self.embed_dim) + support_g_encoder) / 2
        

        support_g_encoder_wei = F.softmax(torch.matmul(query_g.unsqueeze(1), support_g_encoder.transpose(0,1)), dim = -1)
         
        support_g_encoder_expand = support_g_encoder.repeat(batch, 1, 1)
        support_g_encoder_agg = torch.sum(support_g_encoder_wei.transpose(1,2) * support_g_encoder_expand, dim = 1)

        scalar = support_g_encoder.size(1) ** -0.5  # dim ** -0.5
        matching_scores = torch.sum(query_g * support_g_encoder_agg, dim=-1) * scalar

        return matching_scores

