#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from model import BERT
from dataset import BERTDataset
from trainer import BERTTrainer
from corpus_vocab import vocab_file
from tokenizer import BehaviorsBERTTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--max_seq_len", type=int, choices=[64, 128, 256, 512], required=True)
parser.add_argument("--per_gpu_batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--layers", type=int, default=8)
parser.add_argument("--attn_heads", type=int, default=8)
parser.add_argument("--rdn_seed", type=int, default=123456789)
parser.add_argument("--corpus_lines", type=int, default=24381321)
args = parser.parse_args()

max_seq_len = args.max_seq_len
per_gpu_batch_size = args.per_gpu_batch_size
epochs = args.epochs
hidden = args.hidden
layers = args.layers
attn_heads = args.attn_heads
rdn_seed = args.rdn_seed
params_str = "_{}_{}_{}_{}_{}".format(max_seq_len, hidden, layers, attn_heads, rdn_seed)


corpus_path = "/GameBERT/dataset/corpus"
corpus_lines = args.corpus_lines

output_dir = 'bert'
output_path = output_dir + "/model" + params_str
tensorboard_log_dir = output_dir + '/tensorboard' + params_str


def main():

    random.seed(rdn_seed)
    np.random.seed(rdn_seed)
    torch.manual_seed(rdn_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {}, n_gpu: {}".format(device, n_gpu))
    if device == "cuda":
        torch.cuda.manual_seed_all(rdn_seed)

    tokenizer = BehaviorsBERTTokenizer(vocab_file)
    print("Vocab size:", tokenizer.vocab_size)

    train_dataset = BERTDataset(corpus_path, tokenizer, max_seq_len, corpus_lines=corpus_lines)
    batch_size = per_gpu_batch_size * n_gpu
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    bert = BERT(vocab_size=tokenizer.vocab_size,
                hidden=hidden, n_layers=layers,
                attn_heads=attn_heads, max_seq_len=max_seq_len)
    trainer = BERTTrainer(bert, tokenizer.vocab_size, epochs,
                          tensorboard_log_dir=tensorboard_log_dir,
                          output_path=output_path,
                          train_dataloader=train_dataloader,
                          with_cuda=torch.cuda.is_available(),
                          log_freq=100,
                          save_steps=100000)

    trainer.train()


if __name__ == "__main__":
    main()



