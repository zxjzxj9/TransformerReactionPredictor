#! /usr/bin/env python

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEmbedding(nn.Module):

    def __init__(self, maxlen, ndim):
        super().__init__()
        self.embed = nn.Embedding(maxlen, ndim, padding_idx=0)

    def forward(self, x):
        return self.embed(x)

class SinEmbedding(nn.Module):
    
    def __init__(self, maxlen, ndim):
        super().__init__()

        pe = torch.zeros(maxlen, ndim)
        pos = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, ndim, 2)).float()*(-math.log(10000.0)/ndim)
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


class TRPModel(nn.Module):
    
    def __init__(self, nfeat, nlayers, nff, act_fn='relu'):
        super().__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    pass
