import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import math

class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len
        self.d_model = d_model

        self.flinear = nn.Linear(seg_len // 2 + 1, d_model // 2 + 1, bias = False)
        self.linear = nn.Linear(seg_len, d_model, bias = False)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)
        # x_segment = torch.fft.rfft(x_segment, dim = -1).real
        x_embed = self.linear(x_segment)
        # x_embed = torch.fft.irfft(x_embed, n = self.d_model, dim = -1).real
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)
        
        return x_embed