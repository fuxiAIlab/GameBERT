#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from corpus_vocab import vocab_file

from tokenizer import BehaviorsBERTTokenizer
from model import BERTLSS
from dataset import BERTDatasetLSS
from trainer import BERTTrainerLSS

per_gpu_batch_size = 8
epochs = 500

corpus_path = "/GameBERT/dataset/nsh_huge_corpus"
corpus_lines = 24381321
rdn_seed = 123456789
output_dir = 'bert-with-longSequenceSupport'


max_seq_len = 1024
hidden = 256
layers = 8
attn_heads = 8

lr = 1e-4
weight_decay = 0.001
lambda_beta = 1
warmup_steps = 10000
params_str = "_{}_{}_{}_{}_{}_{}_{}_{}".format(max_seq_len, hidden, layers, attn_heads,
                                               lr, weight_decay, lambda_beta, warmup_steps)

tensorboard_log_dir = output_dir + '/tensorboard' + params_str


def construct_pos_mask(stride=32):
    # we use the nearest stride's neighbors around i
    width = stride

    pos_mask_bool = np.zeros([max_seq_len, max_seq_len], dtype=bool)  # [T, T]
    pos_mask = np.zeros([max_seq_len, width, max_seq_len])  # [T, Width, T]
    for i in range(max_seq_len):
        for j in range(max_seq_len):
            if i - width // 2 <= j <= i + width // 2 - 1:
                pos_mask_bool[i, j] = True

        cur_pos_bool = pos_mask_bool[i, :]
        for jj, each in enumerate(np.where(cur_pos_bool)[0]):
            pos_mask[i, jj, each] = 1.0
    pos_mask = torch.tensor(pos_mask, dtype=torch.float)
    return pos_mask   # [T, width, T]


def main():

    output_path = output_dir + "/model" + params_str

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

    train_dataset = BERTDatasetLSS(corpus_path, tokenizer, max_seq_len, corpus_lines=corpus_lines)
    batch_size = per_gpu_batch_size * n_gpu if n_gpu > 0 else 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    bert = BERTLSS(tokenizer.vocab_size, max_seq_len=max_seq_len,
                   hidden=hidden, n_layers=layers, attn_heads=attn_heads)
    bert = bert.to(device)
    trainer = BERTTrainerLSS(bert, tokenizer.vocab_size, epochs,
                             tensorboard_log_dir=tensorboard_log_dir,
                             output_path=output_path,
                             train_dataloader=train_dataloader,
                             with_cuda=torch.cuda.is_available(),
                             log_freq=100,
                             save_steps=100000,
                             lr=lr,
                             weight_decay=weight_decay,
                             warmup_steps=warmup_steps)
    # input("before load position mask to gpu")
    pos_mask = construct_pos_mask().to(device)
    if n_gpu == 1:
        pass
    else:
        pos_mask = pos_mask.unsqueeze(0).repeat([n_gpu, 1, 1, 1])
    # input("after load position mask to gpu")  # 747MiB
    trainer.train(pos_mask)


if __name__ == "__main__":
    main()



