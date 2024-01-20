# -*- coding: utf-8 -*-
# @Time    : 2024/1/20 10:35
# @Author  : yblir
# @File    : swin_transformer.py
# explain  : 
# =======================================================
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_embedding = nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # [n,3,img_h,img_w]->[n,embedding_dim,h,w]
        x = self.patch_embedding(x)
        # [n,embedding_dim,h,w] ->[n,embedding_dim,h*w]
        x = torch.flatten(x, start_dim=2)
        # n,embedding,h*w -> n,h*w,embedding_dim
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        # 将merge后的4倍通道数转为2倍
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        # norm的维度与nn.Linear的输入维度对齐
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        # x.shape=(n,h*w,c)
        b, _, c = x.shape
        x = x.reshape(b, h, w, c)

        # 将每个win窗口拍平，通道维度拼接，使用yolov5系列中的focus操作可实现
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 0::2, 1::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # n,h,w,c -> n,h/2,w/w,4*c
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        # ->b,h*w,4*c
        x = x.reshape(b, -1, 4 * c)

        # norm与linear先后次序对性能影响不同
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


def windows_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    # ->b,h//w,w//ws,h,w,c
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(-1, window_size, window_size, c)

    return x


def windows_reverse(windows, window_size, h, w):
    """
    windows_partition的反操作，做完attention后再变换回来
    :param windows:
    :param window_size:
    :param h:
    :param w:
    :return:
    """
    batch_size = int(windows.shape[0] // (h / window_size * w / window_size))
    x = windows.reshape(batch_size, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.reshape(batch_size, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dim_head = dim // num_heads
        self.dim = dim
        self.scale = self.dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def transpose_multi_head(self, x):
        x = x.reshape(x.shape[:-1] + (self.num_heads, self.dim_head))
        # b,num_heads,num_patchs,dim_head
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def forward(self, x):
        b, n, c = x.shape
        # x:b,num_patchs,embed_dim
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.transpose_multi_head, qkv)

        q = q * self.scale
        # todo 这样转置对吗
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = self.softmax(attn)

        # ->b,num_heads,num_patchs,dim_head
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(b, n, c)
        out = self.proj(out)

        return out


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size):
        super(SwinBlock, self).__init__()
        self.dim = dim
        self.resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads

        self.attention = WindowAttention(dim, window_size, num_heads)
        self.atten_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x):
        h, w = self.resolution
        b, n, c = x.shape

        identity = x
        x = self.atten_norm(x)
        x = x.reshape(b, h, w, c)
        x_windows = windows_partition(x, self.window_size)
        # b*num_patchs,window_size,window_size,c
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, c)
        attention_windows = self.attention(x_windows)
        attention_windows = attention_windows.reshape(-1, self.window_size, self.window_size, c)
        x = windows_reverse(attention_windows, self.window_size, h, w)

        x = x.reshape(b, h * w, c)
        x = identity + x

        identity = x
        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = identity + x

        return x


def main():
    t = torch.randn(4, 3, 224, 224)
    patch_embed = PatchEmbedding(patch_size=4, embedding_dim=96)
    swin_block = SwinBlock(dim=96, input_resolution=[56, 56], num_heads=4, window_size=7)
    patch_merge = PatchMerging(input_resolution=[56, 56], dim=96)

    #
    out = patch_embed(t)
    print("shape=", out.shape)

    out = swin_block(out)
    print("out.shape=", out.shape)

    out = patch_merge(out)
    print("out2.shape=", out.shape)


if __name__ == '__main__':
    main()
