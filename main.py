#! /usr/bin/env python

import os
import sys
import argparse

parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type="str", default="params.yaml", help="Config file path")

arg = parser.parse_args()

def train():
    pass

def predict():
    pass

if __name__ == "__main__":
    pass