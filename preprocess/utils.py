#! /usr/bin/env python

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
            "<UNK>": 0,
            "<GO>": 1,
            "<EOS>": 2,
            "<PAD>": 3,
        }

        self.vdict = {}
        self.idict = {}
    
    def add(self, word):
        self.vocab[word] = 1 \
            if word not in self.vocab \
            else self.vocab[word] + 1
    
    def build(self):
        pass

    def idx_to_token(self, idx):
        pass

    def token_to_idx(self, token):
        if token not in self.vocab:
            raise RuntimeError("Token Lookup Error: No such token {}".format(token))

