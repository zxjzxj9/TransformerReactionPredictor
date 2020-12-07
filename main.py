#! /usr/bin/env python

import os
import sys
import argparse
from .utils import Config, \
    create_model_from_config, \
    create_dateset_from_config, \
    create_optimizer_from_config
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import apex

parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type="str", default="params.yaml", help="Config file path")
parser.add_argument("-m", "--mode", type="str", default="train", help="Training mode")
args = parser.parse_args()

def train(model, optimizer, nepoch, train_data, valid_data, test_data, log_writer=None):
    for src, tgt in tqdm.tqdm(train_data):
        pred = model(src)
        loss = F.binary_cross_entropy_with_logits(pred, tgt)

        optimizer.zero_grad()
        if hasattr(optimizer, "scale_loss"):
            with optimizer.scale_loss(loss) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

def predict():
    pass

if __name__ == "__main__":
    conf = Config(args.config_file)
    train_data, valid_data, test_data = create_dateset_from_config(conf)
    model, vocab = create_model_from_config(conf)
    model, optimizer = create_optimizer_from_config(conf, model)
    
    if args.mode == "train":
        for nepoch in conf["nepochs"]:
            train(model, optimizer, nepoch, train_data, valid_data, test_data)
    elif args.mode == "predict":
        pass
    else:
        print("Invalid mode argument: {}, exiting...".format(args.mode))

