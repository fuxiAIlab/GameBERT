#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tqdm
import os
from torch.utils.data import Dataset
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_len, corpus_lines=None):
        self.corpus_lines = 0
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.blocks = sorted([os.path.join(corpus_path, f) for f in os.listdir(corpus_path)])
        self.n_blocks = len(self.blocks)
        if corpus_lines is None:
            for block in tqdm.tqdm(self.blocks, desc="Total %s blocks" % self.n_blocks, disable=False):
                with open(block, 'r') as fin:
                    for _ in tqdm.tqdm(fin, desc="load block %s" % block, disable=False):
                        self.corpus_lines += 1
        else:
            self.corpus_lines = corpus_lines
        self.block_idx = 0
        self.block_file_handlers = []
        for block in self.blocks:
            self.block_file_handlers.append([open(block, 'r'), block])

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.random_sentence()
        t1 = self.tokenizer.convert_tokens_to_ids(t1)
        t1_random, t1_label = self.random_word(t1)

        t1 = [self.tokenizer.sos_idx] + t1_random + [self.tokenizer.eos_idx]
        t1_label = [self.tokenizer.padding_idx] + t1_label + [self.tokenizer.padding_idx]

        bert_input = t1[:self.max_seq_len]
        bert_label = t1_label[:self.max_seq_len]

        padding = [self.tokenizer.padding_idx for _ in range(self.max_seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def __get_one_line(self):
        while True:
            if self.block_idx == len(self.block_file_handlers):
                self.block_idx = 0

            line = self.block_file_handlers[self.block_idx][0].readline().strip()
            if line == '': # reach the end of file
                self.block_file_handlers[self.block_idx][0].close()
                self.block_file_handlers[self.block_idx][0] = open(
                    self.block_file_handlers[self.block_idx][1], 'r'
                )
                self.block_idx += 1
            else:
                try:
                    line = line.split(':')[1].split(' ')
                except:
                    print(self.block_idx, self.block_file_handlers[self.block_idx][1], line[:50])

                if len(line) > 1:
                    return line

    def random_sentence(self):
        tokens = self.__get_one_line()
        start = random.randint(0, max(len(tokens) - (self.max_seq_len - 2), 0))  # <SOS> and <EOS> tokens
        return tokens[start: start + self.max_seq_len - 2]

    def random_word(self, words):
        tokens = words
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.8:
                    # 80% randomly masked
                    tokens[i] = self.tokenizer.mask_idx
                elif prob < 0.9:
                    # 10% randomly changed to random token
                    tokens[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                else:
                    # 10% randomly remain current token
                    pass
                output_label.append(token)
            else:
                # 85% unprocessed
                output_label.append(0)
        return tokens, output_label
