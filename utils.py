#! /usr/bin/env python

import yaml
from yaml import CSafeLoader
import models
from preprocess import Vocab, USPatent
from models import TRPModel
import pickle
# from bleu import list_bleu

class Config:
    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        with open(folder, "r") as f:
            self.config = yaml.load(f, Loader=CSafeLoader)

    def __getitem__(self, key):
        return self.config[key]

def create_model_from_config(config):
    model_config = config["model_config"]
    data_config = config["data_config"]

    with open(data_config["vocab_file"], "rb") as f:
        vocab = pickle.load(f)

    model = TRPModel(len(vocab), model_config["nfeat"], 
                     model_config["nhead"], model_config["nlayer"],
                     model_config["nff"], model_config["max_len"],
                     model_config["dropout"], model_config["act_fn"])

    return vocab, model

def create_dateset_from_config(config):
    train_data = USPatent(config["train_file"])
    valid_data = USPatent(config["valid_file"])
    test_data = USPatent(config["test_file"])
    
    return train_data, valid_data, test_data


if __name__ == "__main__":
    config = Config("./params.yaml")
    print(create_model_from_config(config))