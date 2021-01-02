#! /usr/bin/env python

import yaml
from yaml import CSafeLoader, Loader
# import models
from preprocess import Vocab, USPatent
from models import TRPModel
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from apex import amp
# from bleu import list_bleu

class Config:
    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        with open(folder, "r") as f:
            self.config = yaml.load(f, Loader=Loader)

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
        optimizer = optim.Adam(model.parameters(), lr=float(train_config["lr"]))
    else:
        raise RuntimeError("Unsupported error: {}"\
            .format(train_config["optimizer"]))

    if train_config["use_fp16"]:
        model, optimizer = amp.initialize(model, optimizer,
            opt_level=train_config["opt_level"])
            
    return model, optimizer

def save_checkpoints(model, path, optimizer=None, niters=0):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer,
        "amp": amp,
        "niters": niters,
    }, path)

def load_checkpoints(model, path, optimizer=None):
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state["model"])
    if optimizer: optimizer.load_state_dict(state["optimizer"])
    if state["amp"] is not None: amp.load_state_dict(state["amp"])
    return model, optimizer

def collate_fn(data):
    print(len(data))
    # import sys; sys.exit()
    src, tgt = zip(*data)
    src = torch.stack(src, dim=1)
    tgt = torch.stack(tgt, dim=1)
    # print(src)
    # print(src, tgt)
    return src, tgt

def create_dateset_from_config(config):
    train_config = config["train_config"]
    data_config = config["data_config"]
    train_data = USPatent(data_config["train_file"])
    valid_data = USPatent(data_config["valid_file"])
    test_data = USPatent(data_config["test_file"])
    
    train_data = DataLoader(train_data, batch_size=train_config["batch_size"], 
        shuffle=True, collate_fn=collate_fn, num_workers=train_config["nworkers"], pin_memory=True)
    valid_data = DataLoader(train_data, batch_size=train_config["batch_size"], 
        shuffle=False, collate_fn=collate_fn, num_workers=train_config["nworkers"], pin_memory=True)
    test_data = DataLoader(train_data, batch_size=train_config["batch_size"], 
        shuffle=False, collate_fn=collate_fn, num_workers=train_config["nworkers"], pin_memory=True)
        
    return train_data, valid_data, test_data


if __name__ == "__main__":
    config = Config("./params.yaml")
    print(create_model_from_config(config))