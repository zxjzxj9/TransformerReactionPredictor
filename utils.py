#! /usr/bin/env python

import yaml

class Config:

    def __init__(self, folder):
        super().__init__()

        self.folder = folder
        with open(folder, "r") as f:
            self.config = yaml.load(f)

    def __getitem__(self, key):
        return self.config[key]