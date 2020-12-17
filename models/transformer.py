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
    
    def __init__(self, ntokens, nfeat, nhead, nlayer, nff, maxlen=64, dropout=0.1, act_fn='relu', beam_size=-1):
        super().__init__()

        self.token_embed = nn.Embedding(ntokens, nfeat)
        self.src_pos_embed = PosEmbedding(maxlen, nfeat)
        self.tgt_pos_embed = PosEmbedding(maxlen, nfeat)

        self.beam_size = beam_size

        encoder_layer = nn.TransformerEncoderLayer(nfeat, nhead, nff, dropout, act_fn)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayer, nn.LayerNorm(nfeat))
        decoder_layer = nn.TransformerDecoderLayer(nfeat, nhead, nff, dropout, act_fn)
        self.decoder = nn.TransformerDecoder(decoder_layer, nlayer, nn.LayerNorm(nfeat))

    def forward(self, src, src_mask, tgt = None, tgt_mask = None, max_step=64):
        
        if self.training:
            src_token_feat = self.token_embed(src) # src SxN
            tgt_token_feat = self.token_embed(tgt) # tgt SxN
            sz, nb = src.size() 
            idx = torch.arange(sz).unsqueeze(-1).to(src.device) # Sx1
            src_pos_feat = self.src_pos_embed(idx).to(src.device)
            tgt_pos_feat = self.tgt_pos_embed(idx).to(tgt.device)
            src_feat = src_token_feat + src_pos_feat # add two embeddings
            tgt_feat = tgt_token_feat + tgt_pos_feat 
            memory = self.encoder(src_feat, src_key_padding_mask = src_mask)
            feat = self.decoder(tgt_feat, memory, tgt_key_padding_mask = tgt_mask, 
                memory_key_padding_mask = src_mask)
            return feat
        else:
            # one decode once
            src_token_feat = self.token_embed(src) # src SxN
            # if beam size is -1, then use greedy decode
            # else use beam decoder
            sz, nb = src.size() 
            idx = torch.arange(sz).unsqueeze(-1).to(src.device) # Sx1
            src_pos_feat = self.src_pos_embed(idx)
            src_feat = src_token_feat + src_pos_feat # add two embeddings
            memory = self.encoder(src_feat, src_key_padding_mask = src_mask)
            if self.beam_size == -1:
                # Do greedy decode, suppose src, dst on same device
                idx = torch.arange(max_step).unsqueeze(-1).to(src.device) # Sx1
                tgt_pos_feat = self.tgt_pos_embed(idx)
                tgt = torch.zeros(max_step, nb).to(src.device)
                tgt[0, :] = 4 # <GO> is #4 token
                for step in range(1, max_step):
                    tgt_feat = self.token_embed(tgt) + tgt_pos_feat
                    tgt_mask = (tgt > 0).transpose(0, 1)
                    feat = self.decoder(tgt_feat, memory, tgt_key_padding_mask = tgt_mask, 
                        memory_key_padding_mask = src_mask) # SxNxF
                    # current step input is the max of the last step
                    tgt[step, :] = feat[step - 1, :, :].argmax(-1)
                return tgt
            else:
                # Do beam decode, suppose the batch_size == 1
                sz, nb = src.size() 
                if nb != 1: raise RuntimeError("For Beam search, batch size must equal to 1")
                src = src.repeat(1, self.beam_size)
                log_prob = torch.zeros(self.beam_size).to(src.device)
                tgt = torch.zeros(max_step, nb).to(src.device)
                tgt[0, :] = 4 # <GO> is #4 token

                for step in range(1, max_step):
                    tgt_feat = self.token_embed(tgt) + tgt_pos_feat
                    tgt_mask = (tgt > 0).transpose(0, 1)
                    feat = self.decoder(tgt_feat, memory, tgt_key_padding_mask = tgt_mask, 
                        memory_key_padding_mask = src_mask) # SxNxF
                    # current step input is the max of the last step
                    tgt[step, :] = feat[step - 1, :, :].argmax(-1)
                    # need to add codes to deal with logprob






if __name__ == "__main__":
    model = TRPModel(345, 256, 8, 3, 1024)
    src = torch.randint(0, 344, (64, 8))
    tgt = torch.randint(0, 344, (64, 8))
    # print(src, tgt)
    src_mask = (src > 0).t()
    tgt_mask = (tgt > 0).t()
    pred = model(src, src_mask, tgt, tgt_mask)
    print(pred.shape)