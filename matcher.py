import logging

import torch
from torch import nn
from torch.nn import init
from modules import *
import torch.nn.functional as F

# 注：relation_id没有train、dev和test中的关系的id
# 这里邻域编码器只是聚合实体邻域的关系
class EmbedMatcher(nn.Module):
    def __init__(self, embed_dim, num_rels, num_ents, rel_embed=None, ent_embed=None, dropout=0.2, batch_size=64, use_pretrain=True, finetune=True, dataset='NELL',beta=5.0):
        super(EmbedMatcher, self).__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.rel_emb = nn.Embedding(num_rels + 1, embed_dim, padding_idx=num_rels) 
        self.ent_emb = nn.Embedding(num_ents + 1, embed_dim, padding_idx=num_ents)
        self.dataset = dataset

        # 关系层参数配置
        self.gcn_w = nn.Linear(2*self.embed_dim, self.embed_dim)             # 权重参数   nn.Linear(输入维度，输出维度) 默认偏差开启
        self.gcn_p = nn.Linear(self.embed_dim, 1)

        # 实体层参数配置
        self.gcn_w2 = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.gcn_q = nn.Linear(self.embed_dim, 1)

        # aotoencoder 自动编码器 配置参数
        self.set_LSTM_encoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional=False)   # nn.LSTM(输入维度，隐藏层维度，循环神经网络层层数)
        self.set_LSTM_decoder = nn.LSTM(2 * self.embed_dim, 2 * self.embed_dim, 1, bidirectional=False)


        self.leak = nn.LeakyReLU(0.2)             #0.3好像最好
        self.dropout = nn.Dropout(dropout)


        init.xavier_normal_(self.gcn_w.weight)     
        init.xavier_normal_(self.gcn_p.weight)
        init.xavier_normal_(self.gcn_w2.weight)
        init.xavier_normal_(self.gcn_q.weight)


        # init.constant_(self.gcn_b, 0)             # 初始化为gcn_b为常量为0
        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.rel_emb.weight.data.copy_(torch.from_numpy(rel_embed)) 
            self.ent_emb.weight.data.copy_(torch.from_numpy(ent_embed))
            # emb_np = np.loadtxt(embed_path)
            #self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))  # 将预训练过的嵌入向量输给nn.Embedding
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.ent_emb.weight.requires_grad = False
                self.rel_emb.weight.requires_grad = False

        d_model = self.embed_dim * 2
        self.support_encoder = SupportEncoder(d_model, 2*d_model, dropout)
        #self.query_encoder = QueryEncoder(d_model, process_steps)
        #self.count = 0
        #self.current_rel = None
        #self.attention_record = open('AArecord.txt','w')

    def neighbor_encoder(self, head_id, connections, num_neighbors):
        num_neighbors = num_neighbors.unsqueeze(1)            # 增加一个维度，方便如果后面计算加权和的平均，如果用不到也可以删去这段代码
        relations = connections[:, :, 0].squeeze(-1)          # [batch,最大邻居数]batch_size
        entities = connections[:, :, 1].squeeze(-1)           # [batch,最大邻居数]

        # 头实体扩展成关系、尾实体的格式一样
        head_id = head_id.unsqueeze(0).transpose(0,1)
        head_id_stack = head_id.expand(head_id.size()[0], relations.size()[1])

        # 转换为嵌入
        head_embed = self.dropout(self.ent_emb(head_id_stack))     # 需要复制多个头实体凑成
        rel_embeds = self.dropout(self.rel_emb(relations))         # (batch, 200, embed_dim*embed_dim)
        ent_embeds = self.dropout(self.ent_emb(entities))          # (batch, 200, embed_dim)    200是最大邻居数，默认长度

        # 拼接计算第一层注意力：关系层注意力->拼接头实体和关系
        concat_embeds = torch.cat((head_embed, rel_embeds), dim=-1)  # 输入是一个矩阵  邻居数 * 2dim
        a_hr = self.gcn_w(concat_embeds)
        #weight_hr = F.softmax(self.leak((self.gcn_p(a_hr).squeeze(-1)).masked_fill_(paddingMatrix, -1e9)), dim=-1)   # 加上padding矩阵
        weight_hr = F.softmax(self.leak(self.gcn_p(a_hr).squeeze(-1)), dim=-1)

        # 拼接计算第二层注意力：实体层注意力->拼接a_hr和尾实体
        concat_embeds1 = torch.cat((a_hr, ent_embeds),dim=-1)
        b_hrt = self.gcn_w2(concat_embeds1)
        weight_hrt = F.softmax(self.leak(self.gcn_q(b_hrt).squeeze(-1)),dim=-1)

        # 第三层注意力：三元组注意力 padding权重消去,若要修改模型这一部分也可以利用上
        triple_weight = weight_hr * weight_hrt                                        # [bacth, 最大邻居数]
        triple_weight = triple_weight.unsqueeze(2)

        head_aggregate = torch.sum(triple_weight * b_hrt, dim=1)

        # 残差再次聚合
        head_aggregate_add = head_aggregate + self.ent_emb(head_id).squeeze(1)
        head_aggregate_mul = self.ent_emb(head_id).squeeze(1) * head_aggregate

        head_combination = (self.leak(head_aggregate_add) + self.leak(head_aggregate_mul))/2
        # 多注意头网络
        # multihead_combination = self.multiHeadAttention(q=head_combination, k=head_combination, v=head_combination)

        return head_combination

    def forward(self,iseval, query, support, query_meta=None, support_meta=None):

        '''
        query: (batch_size, 2)
        support: (few, 2)    few=3
        return: (batch_size, )
        
        if self.current_rel!=None and self.current_rel != current_rel:
            self.attention_record.write('-----------------------------------------------------------------')
            self.attention_record.write(current_rel+'\n')
            self.attention_record.write('support pairs\n')
            for entity_pair in support:
                self.attention_record.write(str(entity_pair.cpu().numpy())+'----'+str(entity_pair[1].cpu().numpy())+'\n')
            self.attention_record.write('positive query+\n')
            self.attention_record.write(str(query[0][0])+'----'+str(query[0][1])+'\n')
            self.attention_record.write('nagetive query+\n')
            self.attention_record.write(str(query[1][0])+'----'+str(query[1][1])+'\n')
        '''
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
        

        # [batch,1,dim] * [few,dim].transpose = [batch,1,few]
        support_g_encoder_wei = F.softmax(torch.matmul(query_g.unsqueeze(1), support_g_encoder.transpose(0,1)), dim = -1)
        '''
        if self.current_rel!=None and self.current_rel != current_rel:
            self.attention_record.write('support and positive query attention\n')
            for i in range(few):
                self.attention_record.write(str(support_g_encoder_wei[0][0][i])+'\n')
            self.attention_record.write('\n')
            self.attention_record.write('support and negative query attention\n')
            for i in range(few):
                self.attention_record.write(str(support_g_encoder_wei[1][0][i])+'\n')
        self.current_rel = current_rel
        '''
        # [bacth,few,dim]
        support_g_encoder_expand = support_g_encoder.repeat(batch, 1, 1)
        # [batch,dim] = [batch,few,1] * [batch,few,dim] 注： 乘法会自动扩展 
        support_g_encoder_agg = torch.sum(support_g_encoder_wei.transpose(1,2) * support_g_encoder_expand, dim = 1)

        scalar = support_g_encoder.size(1) ** -0.5  # dim ** -0.5
        matching_scores = torch.sum(query_g * support_g_encoder_agg, dim=-1) * scalar

        return matching_scores

