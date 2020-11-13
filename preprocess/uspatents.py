#! /usr/bin/env python

""" Dataset reader for the .csv file downloaded from 
    https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
"""

import os
import pandas as pd
import tqdm
from torch.utils.data import Dataset
from utils import Vocab

class USPatent(Dataset):
    def __init__(self, datafile, vocab_path="./vocab.pkl"):
        super().__init__()
        self.datafile = datafile
        self.csv = pd.read_csv(datafile, sep='\t', comment='#', 
                               header=0, error_bad_lines=False, 
                               warn_bad_lines=True)\
                    .dropna(subset=["Source", "Target"]) # type: pd.DataFrame

        # self.csv.dropna(subset=["Source", "Target"])
        # print(self.csv)
        # item = self.csv.iloc[3]
        # print(item["Source"])
        # print(item["Target"])
        # print(self.csv.iloc[2])
        self.vocab = Vocab()
        if not os.path.exists(vocab_path):
            print("Vocabulary file not existing, try building one...")
            for _, rec in tqdm.tqdm(self.csv.iterrows(), total=len(self.csv)):
                for v in rec["Source"].split():
                    self.vocab.add(v)
                for v in rec["Target"].split():
                    self.vocab.add(v)
            print("Saving vocabulary to: {}".format(vocab_path))
            self.vocab.save(vocab_path)
        else:
            print("Loading vocabulary from: {}".format(vocab_path))
            self.vocab = Vocab.load(vocab_path)
        print(self.vocab)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        item = self.csv.iloc[idx]
        src = item["Source"]
        tgt = item["Target"]
        

if __name__ == "__main__":
    data = USPatent("~/HDD/ChemicalReaction/US_patents_1976-Sep2016_1product_reactions_train.csv")