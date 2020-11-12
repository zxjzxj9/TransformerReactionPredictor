#! /usr/bin/env python

""" Dataset reader for the .csv file downloaded from 
    https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
"""

import pandas as pd
from torch.utils.data import Dataset


class USPatent(Dataset):
    def __init__(self, datafile):
        super().__init__()
        self.datafile = datafile
        self.csv = pd.read_csv(datafile)
    
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

if __name__ == "__main__":
    data = USPatent("~/HDD/ChemicalReaction/US_patents_1976-Sep2016_1product_reactions_test.csv")