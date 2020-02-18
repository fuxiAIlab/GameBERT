#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from corpus_vocab import vocab_file

from tokenizer import BehaviorsBERTTokenizer
from model import BERTTE
from dataset import BERTDatasetTE
from trainer import BERTTrainerTE

per_gpt_batch_size = 8
epochs = 500

corpus_path = "/GameBERT/dataset/corpus_with_time"
corpus_lines = 24381321
rdn_seed = 123456789
output_dir = 'bert-with-timeEmbedding'


max_seq_len = 512
hidden = 256
layers = 8
attn_heads = 8
expand_dim = 20

lr = 1e-4
weight_decay = 0.001
lambda_beta = 0.2
warmup_steps = 10000

params_str = "_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(max_seq_len, hidden, layers, attn_heads, expand_dim,
                                                  lr, weight_decay, lambda_beta, warmup_steps)

tensorboard_log_dir = output_dir + '/tensorboard' + params_str


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

    train_dataset = BERTDatasetTE(corpus_path, tokenizer, max_seq_len, corpus_lines=corpus_lines)
    batch_size = per_gpt_batch_size * n_gpu if n_gpu > 0 else 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    bert = BERTTE(tokenizer.vocab_size, hidden=hidden, n_layers=layers, attn_heads=attn_heads, expand_dim=expand_dim)
    bert = bert.to(device)
    trainer = BERTTrainerTE(bert, tokenizer.vocab_size, epochs,
                            tensorboard_log_dir=tensorboard_log_dir,
                            output_path=output_path,
                            train_dataloader=train_dataloader,
                            with_cuda=torch.cuda.is_available(),
                            log_freq=100,
                            save_steps=100000,
                            lr=lr,
                            weight_decay=weight_decay,
                            lambda_beta=lambda_beta,
                            warmup_steps=warmup_steps)

    trainer.train()


if __name__ == "__main__":
    main()



