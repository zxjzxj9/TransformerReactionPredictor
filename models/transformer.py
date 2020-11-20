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
    """ Using TRP mode to predict the reaction from src molecule to tgt molecule
        token_embed: token embedding
        pos_embed: positional embedding
        notice PyTorch use SxNxE, TxNxE for source/target layout,
            and NxS, NxT as padding mask layout
    """
    
    def __init__(self, ntokens, nfeat, nhead, nlayer, nff, maxlen=64, dropout=0.1, act_fn='relu'):
        super().__init__()

        self.token_embed = nn.Embedding(ntokens, nfeat)
        self.pos_embed = PosEmbedding(maxlen, nfeat)

        encoder_layer = nn.TransformerEncoderLayer(nfeat, nhead, nff, dropout, act_fn)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayer, nn.LayerNorm(nfeat))
        decoder_layer = nn.TransformerDecoderLayer(nfeat, nhead, nff, dropout, act_fn)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayer, nn.LayerNorm(nfeat))

    def forward(self, src, src_mask, tgt = None, tgt_mask = None):
        
        if self.training:
            src_token_feat = self.token_embed(src)
            src_pos_feat = self.pos_embed(src)
            feat = src_token_feat + src_pos_feat # add two embeddings
            memory = self.encoder(feat, src_key_padding_mask = src_mask)
            feat = self.decoder(tgt, memory, tgt_key_padding_mask = tgt_mask, 
                memory_key_padding_nask = src_mask)
            return feat
        else:
            pass


if __name__ == "__main__":
    model = TRPModel(345, 256, 8, 3, 1024)
    src = torch.randint(4, 345, (64, 8))
    tgt = torch.randint(4, 345, (64, 8))
    src_mask = (src > 0).t()
    tgt_mask = (tgt > 0).t()
    pred = model(src, src_mask, tgt, tgt_mask)
    print(pred.shape)