#! /usr/bin/env python

import pickle

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
            "<UNK>": 1,
            "<GO>": 1,
            "<EOS>": 1,
            "<PAD>": 1,
        }

        self.vdict = {}
        self.idict = {}
    
    def add(self, word):
        self.vocab[word] = 1 \
            if word not in self.vocab \
            else self.vocab[word] + 1
    
    def build(self):
        voc = list(self.vocab.keys())
        self.vdict = zip(voc, range(len(voc)))
        self.idict = zip(range(len(voc)), voc)

    def idx_to_token(self, idx):
        if idx not in self.idict:
            raise RuntimeError("Index Lookup Error: No such index {}".format(idx))
        return self.idict[idx]

    def token_to_idx(self, token):
        if token not in self.vdict:
            raise RuntimeError("Token Lookup Error: No such token {}".format(token))
        return self.vdict[token]

    def pickle(self, path):
        with open(path, "wb", encoding='utf-8') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def unipckle(path):
        with open(path, "rb", encoding='utf-8') as f:
            return pickle.load(f)


if __name__ == "__main__":
    pass