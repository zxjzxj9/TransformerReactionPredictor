#! /usr/bin/env python

import os
import sys
import argparse
from .utils import Config, \
    create_model_from_config, \
    create_dateset_from_config, \
    create_optimizer_from_config, \
    save_checkpoints
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import apex

from torch.utils.tensorboard import SummaryWriter, writer


parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type="str", default="params.yaml", help="Config file path")
parser.add_argument("-m", "--mode", type="str", default="train", help="Training mode")
parser.add_argument("-s", "--summary-folder", type="str", default="./log", help="Summary writer folder")
args = parser.parse_args()

def train(model, optimizer, niter, train_data, valid_data, test_data, summary_writer=None):
    for src, tgt in tqdm.tqdm(train_data):
        niter += 1
        src_mask = (src > 0).t()
        tgt_mask = (tgt > 0).t()
        pred = model(src, src_mask, tgt, tgt_mask)
        loss = F.cross_entropy(pred, tgt, ignore_index=0, reduction='mean') # ignore <pad> token
        optimizer.zero_grad()
        if hasattr(optimizer, "scale_loss"):
            with optimizer.scale_loss(loss) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        summary_writer.add_scalar("loss", loss.item(), niter)
    return niter

def predict():
    pass

if __name__ == "__main__":
    conf = Config(args.config_file)
    train_data, valid_data, test_data = create_dateset_from_config(conf)
    model, vocab = create_model_from_config(conf)
    model, optimizer = create_optimizer_from_config(conf, model)
    writer = SummaryWriter(args.summary_folder)
    
    if args.mode == "train":
        print("Start training mode....")
        niter = 0
        for nepoch in conf["train_config"]["nepochs"]:
            # update iteration number
            niter = train(model, optimizer, niter, train_data, valid_data, test_data, writer)
            if (niter + 1) % conf["train_config"]["nsave"]:
                save_checkpoints(model, "model.pt", optimizer, niter)
    elif args.mode == "predict":
        pass
    else:
        print("Invalid mode argument: {}, exiting...".format(args.mode))