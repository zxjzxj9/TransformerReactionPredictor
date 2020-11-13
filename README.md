# Transformer Reaction Predictor (TRP)
![](./imgs/tryptophan-3d.png)

## 0. Prerequisites 
In order to run this package, you need to install rdkit and PyTorch, as well as some basic dependencies such as pandas, numpy, matplotlib, etc.

## 1. Motivations
This project makes use of Transformers to predict chemical reactions. The underlying idea is similar to the one in [this paper](https://arxiv.org/pdf/1811.02633.pdf). The name of this project can be abbreviated as TRP, so you can expect I put a [tryptophan](https://en.wikipedia.org/wiki/Tryptophan) molecule as the symbol of the project.I've borrowed some idea from [this repo](https://github.com/pschwllr/MolecularTransformer).

The model needs SMILES to predict chemical reactions, to obtain SMILES from chemical structures, see the [JSME Project](https://peter-ertl.com/jsme/JSME_2020-06-11/JSME_test.html).

## 2. Dataset
The dataset used in this project can be found at [this link](https://ibm.box.com/v/ReactionSeq2SeqDataset), which is the data reference by by aforementioned paper.