# -*- coding: utf-8 -*-
# @Time    : 2024/8/2 上午10:29
# @Author  : yblir
# @File    : transformer2.py
# explain  : 
# =======================================================
import torch
from torch import nn
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, enc_vocal_size, dec_vocal_size,
                 enc_embed_dim, dec_embed_dim, num_heads,
                 drop_prob, num_enc_layers, num_dec_layers):
        super().__init__()
        self.enc_embed = nn.Embedding(enc_vocal_size, enc_embed_dim)
        self.dec_embed = nn.Embedding(dec_vocal_size, dec_embed_dim)

        self.position = PositionEmbedding()
        self.fc = nn.Linear(dec_embed_dim, dec_vocal_size)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(enc_embed_dim, enc_embed_dim, num_heads, drop_prob) for _ in range(num_enc_layers)
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(dec_embed_dim, enc_embed_dim, num_heads, drop_prob) for _ in range(num_dec_layers)
        ])

    @staticmethod
    def pad_mask(seq_q, seq_k, pad):
        bs, len_q = seq_q.size()
        _, len_k = seq_k.size()
        # bs,len_k->bs,len_q,lenK
        mask = (seq_k != pad)[:, None, :]
        # bs,1,len_q,len_k
        return mask.expand(bs, len_q, len_k)[:, None, ...]

    @staticmethod
    def causal_mask(seq_q, seq_k):
        bs, len_q = seq_q.size()
        _, len_k = seq_k.size()

        mask = torch.tril(torch.ones((bs, 1, len_q, len_k), dtype=torch.int32))

        return mask

    def forward(self, enc_input, dec_input, pad=0):
        '''
        :param enc_input: bs,seq_len1
        :param dec_input: bs,seq_len2
        :return:
        '''
        # bs,seq_len1,dim1
        enc_tensor = self.enc_embed(enc_input)
        enc_tensor = self.position(enc_tensor)

        dec_tensor = self.dec_embed(dec_input)
        dec_tensor = self.position(dec_tensor)

        #编码
        enc_pad_mask = self.pad_mask(enc_input, enc_input, pad=pad)
        for layer in self.enc_layers:
            enc_tensor = layer(enc_tensor, enc_pad_mask)

        # 解码
        dec_pad_casual_mask = self.pad_mask(dec_input, dec_input, pad=pad) & self.causal_mask(dec_input, dec_input)
        cross_pad_mask = self.pad_mask(dec_input, enc_input, pad=pad)
        for layer in self.dec_layers:
            dec_tensor = layer(dec_tensor, enc_tensor, dec_pad_casual_mask, cross_pad_mask)

        output = self.fc(dec_tensor)

        return output


class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x


class MultiAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, drop_prob):
        super().__init__()
        # 在cross attention中，kv是计算好的值，不应改变
        # kv占主导，就像一个人要了解别人对自己的关注度，要改变自己适应别人，哪有改变别人的道理
        if dim_kv % num_heads != 0:
            raise ValueError('cannot div')
        self.head_dim_kv = dim_kv // num_heads
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv

        self.q = nn.Linear(dim_q, dim_kv, bias=False)
        self.k = nn.Linear(dim_kv, dim_kv, bias=False)
        self.v = nn.Linear(dim_kv, dim_kv, bias=False)

        self.combine_w = nn.Linear(self.num_heads * self.head_dim_kv, dim_q)
        self.norm = nn.LayerNorm(dim_q)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input_q, input_k, input_v, mask=None):
        bs, len_q, dim_q = input_q.shape
        bs, len_k, dim_k = input_k.shape
        residual = input_q
        # bs,len_q,dim_q-> bs,len_q,dim_kv->bs,self.num_heads,len_q,self.head_dim_kv
        Q = self.q(input_q).reshape(bs, len_q, self.num_heads, self.head_dim_kv).transpose(1, 2)
        # bs,self.num_heads,len_k,self.head_dim_kv
        K = self.k(input_k).reshape(bs, len_k, self.num_heads, self.head_dim_kv).transpose(1, 2)
        # bs,self.num_heads,len_k,self.head_dim_kv
        V = self.v(input_v).reshape(bs, len_k, self.num_heads, self.head_dim_kv).transpose(1, 2)
        # bs,self.num_heads,len_q,len_k
        score = Q @ K.transpose(-2, -1) / np.sqrt(self.head_dim_kv)
        if mask is not None:
            score.masked_fill(mask == 0, float('-inf'))

        # bs,self.num_head,len_q,self.head_dim_kv
        value = torch.softmax(score @ V, dim=-1)
        # bs,len_q,self.num_head,self.head_dim_kv
        value = value.transpose(1, 2).reshape(bs, len_q, self.num_heads* self.head_dim_kv)
        value = self.combine_w(value)

        dec_atten = self.norm(residual + value)

        return dec_atten


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, drop_prob):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
                nn.Dropout(drop_prob)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.fc(x)

        return self.norm(residual + x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiAttention(dim_q, dim_kv, num_heads, drop_prob)
        self.ffn = FeedForward(dim_q, dim_q // 2, drop_prob)

    def forward(self, x, mask):
        # x (bs,len_seq,embed_dim)
        x = self.self_attention(x, x, x, mask=mask)
        x = self.ffn(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, drop_prob):
        super().__init__()
        self.self_attention = MultiAttention(dim_q, dim_q, num_heads, drop_prob)
        self.cross_attention = MultiAttention(dim_q, dim_kv, num_heads, drop_prob)
        self.ffn = FeedForward(dim_q, dim_q // 2, drop_prob)

    def forward(self, input_q, input_kv, pad_casual_mask, pad_mask):
        x = self.self_attention(input_q, input_q, input_q, mask=pad_casual_mask)
        x = self.cross_attention(x, input_kv, input_kv, pad_mask)
        x = self.ffn(x)

        return x


if __name__ == '__main__':
    # 输入token, shape =bs,seq_len
    src_input = torch.tensor([[1, 2, 3, 0], [3, 4, 6, 4], [1, 8, 0, 0]])  # 3,4
    tgt_input = torch.tensor([[4, 1, 4, 7, 0], [4, 6, 12, 6, 1], [6, 0, 0, 0, 0]])  # 3,5
    print(src_input.shape)
    print(tgt_input.shape)
    #enc_vocal_size, dec_vocal_size,
                 # enc_embed_dim, dec_embed_dim, num_heads,
                 # drop_prob, num_enc_layers, num_dec_layers
    model = TransformerModel(enc_vocal_size=20, dec_vocal_size=25,
                             enc_embed_dim=16,dec_embed_dim=24, drop_prob=0.1,
                             num_heads=4, num_enc_layers=6, num_dec_layers=6)
    print(model)
    res = model(src_input, tgt_input, 0)
    # print(res)
    # print(res.shape)