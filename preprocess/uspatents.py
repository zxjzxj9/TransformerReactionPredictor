#! /usr/bin/env python

""" Dataset reader for the .csv file downloaded from 
    https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
"""

import os
import pandas as pd
import tqdm
import torch
from torch.utils.data import Dataset
from .utils import Vocab

class USPatent(Dataset):
    """  Class to read the USPatent dataset (1976-2016)
        Dataset property: 
            - Vocabulary ~ 331
            - Src Maxlen: 876 Avglen:  51.8
            - Tgt Maxlen: 514 Avglen: 41.5
            (It suggests most of the reaction is Combination reaction)
    """
    def __init__(self, datafile, vocab_path="./vocab.pkl", maxlen=64):
        super().__init__()
        self.datafile = datafile
        self.csv = pd.read_csv(datafile, sep='\t', comment='#', 
                               header=0, error_bad_lines=False, 
                               warn_bad_lines=True)\
                    .dropna(subset=["Source", "Target"]) # type: pd.DataFrame
        self.vocab = Vocab()
        self.maxlen = maxlen

        if not os.path.exists(vocab_path):
            print("Vocabulary file not existing, try building one...")
            srclen = []
            tgtlen = []
            for _, rec in tqdm.tqdm(self.csv.iterrows(), total=len(self.csv)):
                cnt = 0
                for v in rec["Source"].split():
                    self.vocab.add(v)
                    cnt += 1
                srclen.append(cnt)
                cnt = 0
                for v in rec["Target"].split():
                    self.vocab.add(v)
                    cnt += 1
                tgtlen.append(cnt)
            print(("Summary: \n" + \
                   "SrcLen Max: {:6d}, Avg: {:12.6f}\n" + \
                   "TgtLen Max: {:6d}, Avg: {:12.6f}\n") \
                        .format(max(srclen), sum(srclen)/len(srclen),
                                max(tgtlen), sum(tgtlen)/len(tgtlen)))
            print("Saving vocabulary to: {}".format(vocab_path))
            self.vocab.build()
            self.vocab.save(vocab_path)
        else:
            print("Loading vocabulary from: {}".format(vocab_path))
            self.vocab = Vocab.load(vocab_path) # type: Vocab
        # print(self.vocab.token_to_idx('<PAD>'))
        # print(self.vocab.token_to_idx('>'))
        # print(self.vocab.idx_to_token(0))
        print(self.vocab)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        item = self.csv.iloc[idx]
        src = item["Source"]
        tgt = item["Target"]

        sos = self.vocab.token_to_idx('<SOS>')
        pad = self.vocab.token_to_idx('<PAD>')
        eos = self.vocab.token_to_idx('<EOS>')
        go = self.vocab.token_to_idx('<GO>')

        src_ret = [sos]
        for token in src.split():
            src_ret.append(self.vocab.token_to_idx(token))
        src_ret.append(eos)
        if len(src_ret) < self.maxlen:
            for _ in range(self.maxlen-len(src_ret)): src_ret.append(pad)
        else:
            src_ret = src_ret[:self.maxlen]

        tgt_ret = [go]
        for token in tgt.split():
            tgt_ret.append(self.vocab.token_to_idx(token))
        tgt_ret.append(eos)
        if len(tgt_ret) < self.maxlen:
            for _ in range(self.maxlen-len(tgt_ret)): tgt_ret.append(pad)
        else:
            tgt_ret = tgt_ret[:self.maxlen]

        src_ret = torch.tensor(src_ret)
        tgt_ret = torch.tensor(tgt_ret)
        # print(src_ret.shape, tgt_ret.shape)
        return src_ret, tgt_ret

if __name__ == "__main__":
    data = USPatent("~/HDD/ChemicalReaction/US_patents_1976-Sep2016_1product_reactions_train.csv")
    print(len(data[0][0]), len(data[0][1]))