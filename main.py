#! /usr/bin/env python

import os
import sys
import argparse
from utils import Config, \
    create_model_from_config, \
    create_dateset_from_config, \
    create_optimizer_from_config, \
    save_checkpoints
import tqdm
import torch.nn as nn
import torch.nn.functional as F
# import torch
# import apex

from torch.utils.tensorboard import SummaryWriter, writer


parser = argparse.ArgumentParser("Transformer Reaction Predictor Argument Parser")
parser.add_argument("-c", "--config-file", type=str, default="params.yaml", help="Config file path")
parser.add_argument("-m", "--mode", type=str, default="train", help="Training mode")
parser.add_argument("-s", "--summary-folder", type=str, default="./log", help="Summary writer folder")
args = parser.parse_args()

def train(model, optimizer, niter, train_data, summary_writer=None):
    model.train()
    pbar = tqdm.tqdm(train_data)
    for src, tgt in pbar:
        # print(src, tgt)
        # by default we use GPU
        src = src.to(device='cuda:0')
        tgt = tgt.to(device='cuda:0')

        # print(src)
        # print(tgt)

        niter += 1
        src_mask = (src == 0).t()
        tgt_mask = (tgt == 0).t()
        pred = model(src, src_mask, tgt, tgt_mask)
        # print(pred)
        pred = pred.permute(0, 2, 1).log_softmax(dim=1)
        # print(pred)
        loss = F.nll_loss(pred, tgt, ignore_index=0, reduction='mean') # ignore <pad> token
        # print(loss)
        # import sys; sys.exit()
        optimizer.zero_grad()
        if hasattr(optimizer, "scale_loss"):
            with optimizer.scale_loss(loss) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        pbar.set_description("In iteration: {:06d}, loss: {:12.6f}".format(niter, loss))
        summary_writer.add_scalar("loss", loss.item(), niter)
    print("")
    return niter

def predict(model, test_data, niter):
    model.eval()
    pbar = tqdm.tqdm(test_data)
    avg_loss = 0.0
    tot_sz = 0.0
    for src, tgt in pbar:
        src = src.to(device='cuda:0')
        tgt = tgt.to(device='cuda:0')

        src_mask = (src == 0).t()
        tgt_mask = (tgt == 0).t()

        # need to add precition method
        pred = model(src, src_mask, tgt, tgt_mask)
        # calculate the NLL loss
        loss = F.nll_loss(pred, tgt, ignore_index=0, reduction='sum') # ignore <pad> token
        avg_loss += loss.item()
        tot_sz += src.size(1) # L x N x F
    print("Perplexity: {:12.6}".format(avg_loss/tot_sz))
    summary_writer.add_scalar("perplexity", avg_loss/tot_sz, niter)
    return avg_loss


if __name__ == "__main__":
    conf = Config(args.config_file)
    train_data, valid_data, test_data = create_dateset_from_config(conf)
    model, vocab = create_model_from_config(conf)
    model, optimizer = create_optimizer_from_config(conf, model)
    writer = SummaryWriter(args.summary_folder)
    
    if args.mode == "train":
        print("Start training mode....")
        niter = 0
        for nepoch in range(conf["train_config"]["nepochs"]):
            # update iteration number
            niter = train(model, optimizer, niter, train_data, valid_data, test_data, writer)
            if (niter + 1) % conf["train_config"]["nsave"]:
                save_checkpoints(model, "model.pt", optimizer, niter)
            # do eval at the end of each epoch
            ppl = predict(model, valid_data, nepoch)
    elif args.mode == "predict":
        ppl = predict(model, test_data, 0)
    else:
        print("Invalid mode argument: {}, exiting...".format(args.mode))
