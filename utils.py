#! /usr/bin/env python

from torch._C import ErrorReport
import yaml
from yaml import CSafeLoader
import models
from preprocess import Vocab, USPatent
from models import TRPModel
import pickle
import torch.optim as optim
from apex import amp
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

    return model, vocab

def create_optimizer_from_config(config, model):
    optimizer = None

    train_config = config["train_config"]

    if train_config["use_cuda"]:
        model.cuda()

    if train_config["optimizer"] == "adam":
        optimizer = optim.Adam(lr=train_config["lr"])
    else:
        raise RuntimeError("Unsupported error: {}"\
            .format(train_config["optimizer"]))

    if train_config["use_fp16"]:
        model, optimizer = amp.initialize(model, optimizer,
            opt_level=train_config["opt_level"])
            
    return model, optimizer

def save_checkpoints(model, optimizer=None, niters=0):
    pass

def load_checkpoints(model, optimizer=None):
    pass

def create_dateset_from_config(config):
    train_data = USPatent(config["train_file"])
    valid_data = USPatent(config["valid_file"])
    test_data = USPatent(config["test_file"])
    
    return train_data, valid_data, test_data


if __name__ == "__main__":
    config = Config("./params.yaml")
    print(create_model_from_config(config))