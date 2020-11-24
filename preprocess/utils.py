#! /usr/bin/env python

import pickle
import random

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction, taken from https://github.com/pschwllr/MolecularTransformer
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class Vocab(object):
    def __init__(self):
        self.vocab = {
            "<PAD>": 1,
            "<UNK>": 1,
            "<SOS>": 1,
            "<EOS>": 1,
            "<GO>" : 1,
        }

        self.vdict = {} # type: dict
        self.idict = {} # type: dict
    
    def __len__(self):
        return len(self.vdict)
    
    def add(self, word):
        self.vocab[word] = 1 \
            if word not in self.vocab \
            else self.vocab[word] + 1
    
    def build(self):
        print("Building Dictionary...")
        voc = list(self.vocab.keys())
        self.vdict = dict(zip(voc, range(len(voc))))
        self.idict = dict(zip(range(len(voc)), voc))

    def idx_to_token(self, idx):
        if idx not in self.idict:
            raise RuntimeError("Index Lookup Error: No such index {}".format(idx))
        return self.idict[idx]

    def token_to_idx(self, token):
        if token not in self.vdict:
            raise RuntimeError("Token Lookup Error: No such token {}".format(token))
        return self.vdict[token]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __str__(self):
        s = "Vocabulary size: {}\n".format(len(self.vocab))
        s += "{ \n"
        sample = random.sample(self.vocab.keys(), 5)
        for k in sample:
            s += "    {:12s}:{:06d}\n".format(k, self.vocab[k])
        s += "    ...\n}"
        return s

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    pass