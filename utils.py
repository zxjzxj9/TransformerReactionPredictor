#! /usr/bin/env python

import yaml
import models
from preprocess import Vocab
from models import TRPModel
import pickle

class Config:
    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        with open(folder, "r") as f:
            self.config = yaml.load(f)

    def __getitem__(self, key):
        return self.config[key]

def create_model_from_config(config):
    model_config = config["model_config"]
    data_config = config["data_config"]

    with open(data_config["vocab_file"], "rw") as f:
        vocab = pickle.load(f)

    model = TRPModel(len(vocab), model_config["nfeat"], 
                     model_config["nhead"], model_config["nlayer"],
                     model_config["nff"], model_config["max_len"],
                     model_config["dropout"], model_config["act_fn"])