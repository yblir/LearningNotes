# -*- coding: utf-8 -*-
# @Time    : 2024/1/17 14:37
# @Author  : yblir
# @File    : transformer_module.py
# explain  : 
# =======================================================
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 构建向后掩码张量的函数
def mask(embedding_dim):
    attn_shape = (1, embedding_dim, embedding_dim)
    # 取得上三角矩阵，对称轴都是0
    subsequent_mask_ = np.triu(torch.ones(attn_shape), k=1).astype("uint8")
    # 最后得到的矩阵，中心线及以下都是1
    return torch.from_numpy(1 - subsequent_mask_)


# def attention(query, key, value, mask=None, dropout=None):
#     """
#     这仅是其中一种表示方法
#     query, key, value代表注意力的三个输入矩阵,是输入向量分别与三个可学习的权重矩阵Wq,Wk,Wv相乘获得。
#     query与key做运算获得注意力得分，得分*value就是当前词对其他词的注意力
#     :param query:
#     :param key:
#     :param value:
#     :param mask: 掩码张量
#     :param dropout: 传入的Dropout实例化对象
#     :return:
#     """
#     # 首先将query最后一个维度提取出来，代表的是词嵌入的维度
#     embedding_dim = query.shape[-1]
#     # 在编码当前位置的词时，对所有位置的词分别有多少的注意力
#     sorces = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)
#
#     if mask is not None:
#         # 利用masked_fill方法，将掩码张量和0进行比较，将0替换成非常小的值
#         sorces = sorces.masked_fill(mask == 0, 1e-9)
#
#     # Softmax可以将分数归一化，这样使得分数都是正数并且加起来等于1,量化注意力比重
#     p_attn = F.softmax(sorces, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     # query的注意力表示，最后的维度代表的是词嵌入的维度
#     return torch.matmul(p_attn, value), p_attn


class SelfEmbedding(nn.Module):
    """自定义embedding方法"""

    def __init__(self, dict_length, embedding_dim):
        """
        :param dict_length: 词表长度，forward中输入的是数字化的文本序列，文本序列中每个token索引必须都在nn.Embedding中找到，
                            因此dict_length是整个文本空间不重复token(单词/文字)的数量,如一篇文章中中300个不重复的词汇，len不能少于300
        :param embedding_dim: 向量化后每个token的维度
        """
        super(SelfEmbedding, self).__init__()
        self.embedding = nn.Embedding(dict_length, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        :param x: 通过词汇映射后的数字张量
        :return:
        """
        return self.embedding(x) * math.sqrt(self.embedding_dim)


class PositionalEncoding(nn.Module):
    """位置编码，一脸懵逼！"""

    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        """
        :param embedding_dim: 代表词嵌入的维度
        :param dropout: 代表置零比率
        :param max_len: 每个句子的最大长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，大小是max_len*embedding_dim
        # 句子的每个词都进行位置编码
        position_embedding = torch.zeros(max_len, embedding_dim)

        # 初始化一个绝对位置编码矩阵，shape=(max_len,1)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义一个变换矩阵div_term,跳跃式初始化
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000) / embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0)

        # 将位置编码矩阵注册成模型的buff，这个buff不是模型中的参数，不跟随优化器同步更新
        # 初测成buff后，可以在模型保存后重新加载的时候将这个位置编码器和模型参数加载进来
        # self.register_buffer就是pytorch框架用来保存不更新参数的方法
        self.register_buffer('position_embedding', position_embedding)

    def forward(self, x):
        """
        :param x: 文本序列的词嵌入表示
        :return:
        """
        # 首先明确，pe的编码太长，将第二个维度（max_len）,缩小为句子长度同等的长度
        x = x + Variable(self.position_embedding[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_nums, embedding_dim, dropout=0.1):
        '''

        :param head_nums:代表几个头的参数
        :param embedding_dim: 代表词嵌入的维度
        :param dropout:
        '''
        super(MultiHeadAttention, self).__init__()
        self.head_nums = head_nums

        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        assert embedding_dim % head_nums == 0
        # 获得每个头得到的词向量的维度
        self.d_k = embedding_dim // head_nums

        # 获得线性层,要获得4个, 分别是q,k,v以及最终输出的线性层
        # self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.linears = nn.ModuleList([
            copy.deepcopy(nn.Linear(embedding_dim, embedding_dim))
            for _ in range(4)
        ])

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        这仅是其中一种表示方法
        query, key, value代表注意力的三个输入矩阵,是输入向量分别与三个可学习的权重矩阵Wq,Wk,Wv相乘获得。
        query与key做运算获得注意力得分，得分*value就是当前词对其他词的注意力
        :param query:
        :param key:
        :param value:
        :param mask: 掩码张量
        :param dropout: 传入的Dropout实例化对象
        :return:
        """
        # 首先将query最后一个维度提取出来，代表的是词嵌入的维度
        embedding_dim = query.shape[-1]
        # 在编码当前位置的词时，对所有位置的词分别有多少的注意力
        sorces = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)

        if mask is not None:
            # 利用masked_fill方法，将掩码张量和0进行比较，将0替换成非常小的值
            sorces = sorces.masked_fill(mask == 0, 1e-9)

        # Softmax可以将分数归一化，这样使得分数都是正数并且加起来等于1,量化注意力比重
        p_attn = F.softmax(sorces, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # query的注意力表示，最后的维度代表的是词嵌入的维度
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        """当query == key == value时,成为多头自注意力机制"""
        if mask is not None:
            # 使用squeeze扩充维度,代表多头中的第n个头
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query_, key_, value_ = [
            # self.head:每个词向量被分为几个头(切割为几块),self.d_k:每个头包含向量的长度
            model(x).reshape(batch_size, -1, self.head_nums, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, [query, key, value])
        ]

        # 将每个头的输出传入注意力层
        x, _ = self.attention(query_, key_, value_, mask, self.dropout)
        # 得到的结果是4维tensor,需要进行shape转换
        x = x.transpose(1, 2).contiguous().reshape(batch_size, -1, self.head_nums * self.d_k)

        # 对x在进行一次线性层处理,得到最终的多头注意力结构
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout):
        """
        :param embedding_dim: 词嵌入维度，也是两个线性的输入输出维度
        :param d_ff: 第一个线性层输出，第二个线性层输入维度
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x:来自上一层的输出
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class SubLayerConnection(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.norm(x)))


class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        # dim: 词嵌入维度， eps:一个足够小的整数，方式除0操作
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(embed_dim))
        self.b2 = nn.Parameter(torch.zeros(embed_dim))

        self.eps = eps

    def forward(self, x):
        # 最后一个维度求均值，标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, multi_self_attention, feed_forward, dropout=0.1):
        """
        :param embedding_dim: 词嵌入维度
        :param multi_self_attention: 多头自注意力层实例化对象
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout:
        """
        super(EncoderBlock, self).__init__()
        self.multi_self_attention = multi_self_attention
        self.feed_forward = feed_forward

        # self.sub_layer = nn.ModuleList([
        #     copy.deepcopy(SubLayerConnection(embedding_dim=embedding_dim, dropout=dropout)) for _ in range(2)
        # ])

        self.sub_layer1 = SubLayerConnection(embedding_dim=embedding_dim, dropout=dropout)
        self.sub_layer2 = SubLayerConnection(embedding_dim=embedding_dim, dropout=dropout)

    def forward(self, x, mask):
        """
        :param x: 上一层传入的张量
        :param mask:
        :return:
        """
        # 第一个子层，包含多头自注意力机制
        # lambda... 返回一个匿名函数名，这个函数其实就是def xxx(x): return self.multi_self_attention(x, x, x, mask)
        x = self.sub_layer1(x, lambda x: self.multi_self_attention(x, x, x, mask))

        # 第二个子层，包含前馈全连接层
        return self.sub_layer2(x, self.feed_forward)


class Encoder(nn.Module):
    """构建模型编码器"""
    def __init__(self, layer, N):
        """
        :param layer: 编码器EncoderBlock类对象
        :param N: EncoderBlock重复次数
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # self.n = N  # 代表有几个layer
        # 初始化一个规范化层,作用在编码器最后面
        self.norm = LayerNorm(N)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, multi_self_attention, general_attention, feed_forward, dropout):
        """
        multi_self_attention,general_attention都是由同一个类生成，靠在forward中传入qkv是否相同开区分是自注意力还是常规
        :param embedding_dim: 词向量维度
        :param multi_self_attention: 多头自注意力q=k=v
        :param general_attention: 常规注意力,q!=k=v
        :param feed_forward: 前馈传播网络
        :param dropout:
        """
        super(DecoderBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.multi_self_attention = multi_self_attention
        self.general_attention = general_attention
        self.feed_forward = feed_forward
        self.dropout = dropout

        # 创建连接子层
        self.sub_layer = nn.ModuleList([
            copy.deepcopy(SubLayerConnection(embedding_dim=embedding_dim, dropout=dropout)) for _ in range(3)
        ])

    def forward(self, x, encoder_output, source_mask, target_mask):
        """
        :param x: 上一层输入张量
        :param encoder_output: 编码器最后一层输出,存储的语义张量, 解码器输入的 encoder_output相同
        :param source_mask: 源数据掩码
        :param target_mask: 目标数据掩码
        :return:
        """
        # 1 x 经历第一个子层,多头自注意力机制,使用target_mask, 使解码时看不到当前次以后的词
        x = self.sub_layer[0](x, lambda x: self.multi_self_attention(x, x, x, target_mask))
        # 2 经历常规注意力机制子层 q!=k=v,source_target掩盖掉队结果无用的数据?(不懂)
        x = self.sub_layer[1](x, lambda x: self.general_attention(x, encoder_output, encoder_output, source_mask))
        # 3 前馈全连接层
        return self.sub_layer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """构建模型解码器"""
    def __init__(self, layer, N):
        """
        :param layer: 解码器层DecoderBlock对象
        :param N: DecoderBlock重复次数
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(layer) for _ in range(3)
        ])
        # 实例化一个规范化层
        self.norm = LayerNorm(N)

    def forward(self, x, encoder_output, source_mask, target_mask):
        """
        :param x: 上一层输入张量
        :param encoder_output: 编码器最后一层输出,存储的语义张量, 解码器输入的 encoder_output相同
        :param source_mask: 源数据掩码
        :param target_mask: 目标数据掩码
        :return:
        """
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        return self.norm(x)


class OutputLayer(nn.Module):
    def __init__(self, target_dict_length, embedding_dim):
        """
        target_dict_length 目标文本语库中，所有不重复词的数量
        :param target_dict_length: 线性层输出维度,输出一个词后, 它可以选择文本中所有词概率最大值继续输出
        :param embedding_dim: 词向量维度
        """
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(embedding_dim, target_dict_length)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=-1)


class TransformerModel(nn.Module):
    def __init__(self, encoders, decoders, source_embedding, target_embedding, position, output_layer):
        """
        :param encoders: 编码器模块类对象
        :param decoders: 解码器模块类对象
        :param source_embedding: 源数据embedding模型类对象
        :param target_embedding: 目标数据embedding模型类对象
        :param position: 位置编码类对象
        :param output_layer: 输出层类对象
        """
        super(TransformerModel, self).__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.position = position
        self.output_layer = output_layer

    def forward(self, source_input, target_input, source_mask, target_mask):
        """
        :param source_input: 编码器输入原数据
        :param target_input: 解码器输入原数据
        :param source_mask: 编码器掩码
        :param target_mask: 解码器掩码
        :return:
        """
        # 对已数字化且不连续的输入进行编码，编码后shape=(n,ndim),n: 输入序列长度，ndim:每个token(汉字，单词)的维度
        input_embed = self.source_embedding(source_input)
        input_embed = self.position(input_embed)

        target_embed = self.target_embedding(target_input)
        target_embed = self.position(target_embed)

        # 编码器编码
        encoders_output = self.encoders(input_embed, source_mask)
        # 解码器解码
        target_embedding = self.decoders(target_embed, encoders_output, source_mask, target_mask)
        # 输出层
        return self.output_layer(target_embedding)


def make_transformer_model(source_vocab_length, target_vocab_length,
                           N=6, embedding_dim=512, d_ff=2048, head_nums=8, dropout=0.1):
    """
    传入参数,构建transformer模型
    :param source_vocab_length: 输入源数据不重复的词汇总数
    :param target_vocab_length: 目标数据不重复的词汇总数
    :param N: 编码器和解码器堆叠层数
    :param embedding_dim: 词向量维度
    :param d_ff: 前馈全连接层变换矩阵维度,两个全连接层中间变换维度,fc1的输出维度,fc2的输入维度
    :param head_nums: 多头注意力机制中头数
    :param dropout: 置零比率
    :return:
    """
    encoder_block = EncoderBlock(
            embedding_dim=embedding_dim,
            multi_self_attention=MultiHeadAttention(head_nums, embedding_dim=embedding_dim),
            feed_forward=FeedForward(embedding_dim=embedding_dim, d_ff=d_ff, dropout=dropout),
            dropout=dropout
    )
    encoders = Encoder(encoder_block, N)

    decoder_block = DecoderBlock(
            embedding_dim=embedding_dim,
            # multi_self_attention,general_attention都是MultiHeadAttention对象，靠forward中qkv值区分自注意力还是常规
            multi_self_attention=MultiHeadAttention(head_nums, embedding_dim=embedding_dim),
            general_attention=MultiHeadAttention(head_nums, embedding_dim=embedding_dim),
            feed_forward=FeedForward(embedding_dim=embedding_dim, d_ff=d_ff, dropout=dropout),
            dropout=dropout
    )
    decoders = Decoder(decoder_block, N)

    # 实例化model
    transformer_model = TransformerModel(
            encoders, decoders,
            source_embedding=SelfEmbedding(source_vocab_length, embedding_dim),
            target_embedding=SelfEmbedding(target_vocab_length, embedding_dim),
            position=PositionalEncoding(embedding_dim=embedding_dim, dropout=dropout),
            output_layer=OutputLayer(target_vocab_length, embedding_dim)
    )

    # 初始化模型
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model


if __name__ == '__main__':
    # dict_length_ = 1000
    # embedding_dim_ = 512
    # source_embedding_ = nn.Embedding(dict_length_, embedding_dim_)
    # target_embedding_ = nn.Embedding(dict_length_, embedding_dim_)

    # encoders=Encoder(dict_length)
    source_vocab_length = 11
    target_vocab_length = 11
    N = 6
    res = make_transformer_model(source_vocab_length, target_vocab_length, N)
    print(res)
