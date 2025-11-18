import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import copy
from torch import Tensor
from typing import Optional
from model.gcn import RGCN
from model.crossatt import ResidualCrossAttentionBlock

from torch.nn import functional as F

class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3, stride=1, padding=1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                # nn.LayerNorm(int(outchannel / 1),dim=1)
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        # x = x + self.embedding_1D(torch.arange(52).cuda()).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, att_weight = self._mha_block(self_att_tgt,
                                               memory, memory_mask,
                                               memory_key_padding_mask)
     
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x+tgt
        #return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
 
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x, att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),  att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class StackTransformer(nn.Module):
    r"""StackTransformer is a stack of N decoder layers

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, encoder_dim, feature_dim, vocab_size, max_lengths, word_vocab, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=",n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(encoder_dim*2, feature_dim, kernel_size = 1)
        self.LN = resblock(feature_dim, feature_dim)
        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding
        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()  # initialize some layers with the uniform distribution
        self.fine_proj = nn.Linear(256, 2048)
        print('Loading gcn') 
        self.gcn_proj = nn.Linear(768, 2048)
        self.gcn = RGCN(num_features=768, hidden_dim=384, output_dim=768, dropout=0.1, num_relations=26)
        self.node_features, self.edge_index, self.edge_type = self.load_gcn()
        self.cross_attn_g2i = ResidualCrossAttentionBlock(d_model=2048, n_head=16, dropout=0.1)
        self.cross_attn_i2g = ResidualCrossAttentionBlock(d_model=2048, n_head=16, dropout=0.1)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def load_gcn(self):
        node_features = []
        edge_index = []
        edge_type = []
        # model = torch.load('model.pt')
        node_features = torch.load('/rydata/wangmengqi/Chg2Cap-main/kg_embedding/extracted_node_features.pt')
        edge_index = torch.load('/rydata/wangmengqi/Chg2Cap-main/kg_embedding/edge_index.pt')
        edge_type = torch.load('/rydata/wangmengqi/Chg2Cap-main/kg_embedding/modified_edge_type.pt')
        return node_features, edge_index , edge_type

    # self.node_features, self.edge_index, self.edge_type = self.load_gcn()
    
    def forward(self, x1, x2, encoded_captions, caption_lengths,fineA,fineB,semanticA,semanticB):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param encoded_captions: a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: a tensor of dimension (batch_size)
        """
        
        device = x1.device  # 获取 x1 所在的设备
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim = 1) + x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))

        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1) #x size: torch.Size([64, 32, 2048])
        # print("x size:", x.size()) 
        
        fineA = fineA.to(device)
        fineB = fineB.to(device)
        semanticA=semanticA.to(device)
        semanticB=semanticB.to(device)  
    # 1. 先把 A 的 fine+semantic 拼一起，B 的 fine+semantic 拼一起
    #    fineA, semanticA: (batch, N1, 256)；fineB, semanticB: (batch, N2, 256)
        feats_A = torch.cat([fineA, semanticA], dim=1)  # (batch, N1+N1', 256)
        feats_B = torch.cat([fineB, semanticB], dim=1)  # (batch, N2+N2', 256) 
    # 2. 再把 A 和 B 拼在一起
        combined_feats = torch.cat([feats_A, feats_B], dim=1)  # (batch, N_total, 256)

    # 3. 线性投影到 Transformer 输入维度（假设是 2048）
        combined_feats = self.fine_proj(combined_feats)        # (batch, N_total, 2048)

    # 4. 转成 (sequence_len, batch, feature_dim)
        combined = combined_feats.permute(1, 0, 2)             # (N_total, batch, 2048)
        
        node_features =self.node_features.to(device)
        edge_index =self.edge_index.to(device)
        edge_type =self.edge_type.to(device)
        
        node_features = self.gcn(node_features, edge_index, edge_type).expand(batch, -1, -1)
        node_features = self.gcn_proj(node_features).permute(1, 0, 2)
        
        combined_all=torch.cat([combined, node_features], dim=0)
        # print("Modified node_features:", node_features.size())
        # x = torch.cat([x, node_features], dim=0) 
        image_features_i2g = self.cross_attn_i2g(x, combined_all, combined_all)
        node_features_g2i = self.cross_attn_g2i(combined_all, x, x)     
        x = torch.cat([x, node_features_g2i, image_features_i2g], dim=0) # 再跟原来的输入视觉特征拼接，否则模型忽略了全局特征

        
        word_length = encoded_captions.size(1)
        mask = torch.triu(torch.ones(word_length, word_length) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt_pad_mask = (encoded_captions == self.word_vocab['<NULL>'])|(encoded_captions == self.word_vocab['<END>'])
        word_emb = self.vocab_embedding(encoded_captions) #(batch, length, feature_dim)
        word_emb = word_emb.transpose(1, 0)#(length, batch, feature_dim)

        word_emb = self.position_encoding(word_emb)  # (length, batch, feature_dim)

        pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)  # (length, batch, feature_dim)
        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
        pred = pred.permute(1, 0, 2)
        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        #encoded_caption = torch.cat((encoded_captions, torch.zeros([batch, 1], dtype = int).cuda()), dim=1)
        #decode_lengths = (caption_lengths).tolist()
        return pred, encoded_captions, decode_lengths, sort_ind

    def sample(self, x1, x2, fineA, fineB, semanticA, semanticB, imgA, imgB, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        """
        device = x1.device  # 获取 x1 所在的设备
        x_sam = self.cos(x1, x2)
        x = torch.cat([x1, x2], dim = 1) + x_sam.unsqueeze(1) #(batch_size, 2channel, enc_image_size, enc_image_size)
        x = self.LN(self.Conv1(x))

        batch, channel = x.size(0), x.size(1)
        x = x.view(batch, channel, -1).permute(2, 0, 1) #x size: torch.Size([64, 32, 2048])
        # print("x size:", x.size()) 
        
        fineA = fineA.to(device)
        fineB = fineB.to(device)
        semanticA=semanticA.to(device)
        semanticB=semanticB.to(device)  
    # 1. 先把 A 的 fine+semantic 拼一起，B 的 fine+semantic 拼一起
    #    fineA, semanticA: (batch, N1, 256)；fineB, semanticB: (batch, N2, 256)
        feats_A = torch.cat([fineA, semanticA], dim=1)  # (batch, N1+N1', 256)
        feats_B = torch.cat([fineB, semanticB], dim=1)  # (batch, N2+N2', 256) 
    # 2. 再把 A 和 B 拼在一起
        combined_feats = torch.cat([feats_A, feats_B], dim=1)  # (batch, N_total, 256)

    # 3. 线性投影到 Transformer 输入维度（假设是 2048）
        combined_feats = self.fine_proj(combined_feats)        # (batch, N_total, 2048)

    # 4. 转成 (sequence_len, batch, feature_dim)
        combined = combined_feats.permute(1, 0, 2)             # (N_total, batch, 2048)
        
        node_features =self.node_features.to(device)
        edge_index =self.edge_index.to(device)
        edge_type =self.edge_type.to(device)
        
        node_features = self.gcn(node_features, edge_index, edge_type).expand(batch, -1, -1)
        node_features = self.gcn_proj(node_features).permute(1, 0, 2)
        
        combined_all=torch.cat([combined, node_features], dim=0)
        # print("Modified node_features:", node_features.size())
        # x = torch.cat([x, node_features], dim=0) 
        image_features_i2g = self.cross_attn_i2g(x, combined_all, combined_all)
        node_features_g2i = self.cross_attn_g2i(combined_all, x, x)     
        x = torch.cat([x, node_features_g2i, image_features_i2g], dim=0) # 再跟原来的输入视觉特征拼接，否则模型忽略了全局特征
        
        tgt = torch.zeros(batch, self.max_lengths).to(torch.int64).cuda()

        mask = torch.triu(torch.ones(self.max_lengths, self.max_lengths) * float('-inf'), diagonal=1)
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch).cuda() #(batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch).cuda()
        #Weight = torch.zeros(1, self.max_lengths, x.size(0)).cuda()
        for step in range(self.max_lengths):
            tgt_pad_mask = (tgt == self.word_vocab['<NULL>'])
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)#(length, batch, feature_dim)

            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)

            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2) # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            predicted_id = torch.argmax(scores, axis=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim = -1)
            #Weight = torch.cat([Weight, weight], dim = 0)
            if predicted_id == self.word_vocab['<END>']:
                break
            if step<(self.max_lengths-1):#except <END> node
                tgt[:, step+1] = predicted_id
        seqs = seqs.squeeze(0)
        seqs = seqs.tolist()
        
        #feature=x.clone()
        #Weight1=Weight.clone()
        return seqs


    def sample1(self, x1, x2, k=1):
        """
        :param x1, x2: encoded images, a tensor of dimension (batch_size, channel, enc_image_size, enc_image_size)
        :param max_lengths: maximum length of the generated captions
        :param k: beam_size
        """

        x = torch.cat([x1, x2], dim = 1)
        x = self.LN(self.Conv1(x))
        batch, channel, h, w = x.shape
        x = x.view(batch, channel, -1).unsqueeze(0).expand(k, -1, -1, -1).reshape(batch*k, channel, h*w).permute(2, 0, 1) #(h*w, batch, feature_dim)
        node_features =self.node_features
        edge_index =self.edge_index
        edge_type =self.edge_type
        
        node_features = self.gcn(node_features, edge_index, edge_type).expand(batch, -1, -1)
        node_features = self.gcn_proj(node_features).permute(1, 0, 2)

        x = torch.cat([x, node_features], dim=0) 
        tgt = torch.zeros(k*batch, self.max_lengths).to(torch.int64).cuda()

        mask = (torch.triu(torch.ones(self.max_lengths, self.max_lengths)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.cuda()
        tgt[:, 0] = torch.LongTensor([self.word_vocab['<START>']] *batch*k).cuda() #(batch_size*k, 1)
        seqs = torch.LongTensor([[self.word_vocab['<START>']]] *batch*k).cuda()
        top_k_scores = torch.zeros(k*batch, 1).cuda()
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt)
            word_emb = word_emb.transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)
            scores = pred.permute(1, 0, 2) # (batch, length, vocab_size)
            scores = scores[:, step, :].squeeze(1)  # [batch, 1, vocab_size] -> [batch, vocab_size]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode='floor')
            next_word_inds = top_k_words % self.vocab_size  # (s)
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim = 1)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word_vocab['<END>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:,prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step<self.max_lengths-1:
                tgt[:, :step+2] = seqs


        if complete_seqs == []:
            complete_seqs.extend(seqs[incomplete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[incomplete_inds])
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq
