#! /usr/bin/env python

import os
import sys
import argparse
from .utils import Config, \
    create_model_from_config, \
    create_dateset_from_config, \
    create_optimizer_from_config

parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type="str", default="params.yaml", help="Config file path")
parser.add_argument("-m", "--mode", type="str", default="train", help="Training mode")
args = parser.parse_args()

def train(model, optimizer, niter):
    pass

def predict():
    pass

if __name__ == "__main__":
    conf = Config(args.config_file)
    train_data, valid_data, test_data = create_dateset_from_config(conf)
    model, vocab = create_model_from_config(conf)
    
    if args.mode == "train":
        pass
    elif args.mode == "predict":
        pass
    else:
        print("Invalid mode argument: {}, exiting...".format(args.mode))