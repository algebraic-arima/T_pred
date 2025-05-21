from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class lerrorModel(nn.Module):
    def __init__(self, data_dim, in_len, out_len, device=torch.device('cuda:0')):
        super(lerrorModel, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.device = device
        
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm = nn.LayerNorm(out_len)
        self.linear = nn.Linear(in_len, out_len, bias=True)
        self.linear2 = nn.Linear(in_len - 3, out_len, bias=True)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x_seq):

        x_seq = rearrange(x_seq, 'b l d -> b d l')
        y = self.linear(x_seq)
        y = self.relu(y)
        e = self.linear2(torch.diff(torch.diff(torch.diff(x_seq, dim=2), dim=2)))
        e = self.relu2(e)
        y = self.dropout(y + e)
        y = self.norm(y)
        y = rearrange(y, 'b d l -> b l d')

        return y
    
