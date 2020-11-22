#! /usr/bin/env python

import yaml
from preprocess import Vocab
from models import TRPModel

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

    
