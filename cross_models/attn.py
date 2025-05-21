import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.5):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        # (blhe * bshe^T) * bshd 
        queries = torch.fft.rfft(queries, dim=-2).real
        keys = torch.fft.rfft(keys, dim=-2).real
        values = torch.fft.rfft(values, dim=-2).real
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        V = torch.fft.irfft(V, n=H, dim=-2).real
        return V.contiguous()
    
    def forward2(self, queries, keys, values):
        # implement of linear attention
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        Q = F.elu(queries)+1
        K = F.elu(keys)+1
        A = self.dropout(torch.einsum("bshe,bshd->behd", K, values))
        A = torch.einsum("blhe,behd->blhd", Q, A)
        Z = torch.einsum("blhe,bshe->blhs", Q, K.sum(dim=1, keepdim=True)).repeat(1, 1, 1, D) # s = 1
        V = A / (Z + 1e-6)

    
        return V.contiguous()

    def forward3(self, queries, keys, values):
        # implement of cosformer, Q_i^cos = Q_i*cos(pi*i/2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        M = max(L, S)
        # print(M)
        i = torch.arange(L, device=queries.device).float().view(1, L, 1)
        j = torch.arange(S, device=queries.device).float().view(1, 1, S)
        cs_i = torch.stack([torch.cos(math.pi * i / 2 / M), torch.sin(math.pi * i / 2 / M)], dim=-1).view(1, L, 1, 1, 2)
        cs_j = torch.stack([torch.cos(math.pi * j / 2 / M), torch.sin(math.pi * j / 2 / M)], dim=-1).view(1, S, 1, 1, 2)
        Q_p = repeat(torch.relu(queries), 'b l h e->b l h e u', u=2)
        K_p = repeat(torch.relu(keys), 'b s h e->b s h e u', u=2)
        V_p = repeat(torch.relu(values), 'b s h d->b s h d u', u=2)
        Q_cs = Q_p * cs_i
        K_cs = K_p * cs_j

        KV_cs = torch.einsum("bsheu,bshdu->behdu", K_cs, V_p)

        out_cs = torch.einsum("blheu,behdu->blhdu", Q_cs, KV_cs)
        out_cs = self.dropout(out_cs)

        Z_cs = torch.einsum("blheu,bsheu->blhsu", Q_cs, K_cs.sum(dim=1, keepdim=True)).repeat(1, 1, 1, D, 1)

        a = out_cs.sum(dim=-1) / (Z_cs.sum(dim=-1) + 1e-6)
        
        return a.contiguous()
    


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = out.view(B, L, -1)

        return self.out_projection(out)

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)

        return final_out
