from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFTModel(nn.Module):
    def __init__(self, data_dim, in_len, out_len, device=torch.device('cuda:0')):
        super(FFTModel, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.device = device
        
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.norm = nn.LayerNorm(out_len // 2 + 1)
        self.linear = nn.Linear(in_len, out_len, bias=True)
        self.linear2 = nn.Linear(in_len - 3, out_len, bias=True)

        self.fftlinear = nn.Linear(in_len // 2 + 1, out_len // 2 + 1, bias = False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x_seq):

        x_seq = rearrange(x_seq, 'b l d -> b d l')
        x_freq = torch.fft.rfft(x_seq, dim=2).real
        x_freq = self.fftlinear(x_freq)
        x_freq = self.dropout(x_freq)
        x_freq = self.relu(x_freq)
        x_freq = self.norm(x_freq)
        y = torch.fft.irfft(x_freq, n=self.out_len, dim=2).real
        y = rearrange(y, 'b d l -> b l d')

        return y
    
