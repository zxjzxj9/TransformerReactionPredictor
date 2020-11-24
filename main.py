#! /usr/bin/env python

import os
import sys
import argparse
from .utils import Config, create_model_from_config

parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type="str", default="params.yaml", help="Config file path")
parser.add_argument("-m", "--mode", type="str", default="train", help="Training mode")
args = parser.parse_args()

def train():
    pass 

def predict():
    pass

if __name__ == "__main__":
    conf = Config(args.config_file)
    model, dataset = create_model_from_config(conf)
    
    if args.mode == "train":
        pass
    elif args.mode == "predict":
        pass
    else:
        print("Invalid mode argument: {}, exiting...".format(args.mode))