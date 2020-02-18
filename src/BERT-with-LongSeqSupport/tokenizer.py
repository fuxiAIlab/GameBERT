#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from corpus_vocab import specials, UNK_SYMBOL, PAD_SYMBOL, EOS_SYMBOL, SOS_SYMBOL, MASK_SYMBOL


class BehaviorsBERTTokenizer:
    def __init__(self, vocab_file):
        itos = copy.deepcopy(specials)
        with open(vocab_file, 'r') as fin:
            for l in fin:
                itos.append(l.strip())
        self.stoi = {s: i for i, s in enumerate(itos)}

    def convert_tokens_to_ids(self, token_or_tokens):
        if isinstance(token_or_tokens, list):
            return [self.stoi.get(e, self.stoi[UNK_SYMBOL]) for e in token_or_tokens]
        elif isinstance(token_or_tokens, str):
            return self.stoi.get(token_or_tokens, self.stoi[UNK_SYMBOL])

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def unknown_idx(self):
        return self.stoi[UNK_SYMBOL]

    @property
    def padding_idx(self):
        return self.stoi[PAD_SYMBOL]

    @property
    def sos_idx(self):
        return self.stoi[SOS_SYMBOL]

    @property
    def eos_idx(self):
        return self.stoi[EOS_SYMBOL]

    @property
    def mask_idx(self):
        return self.stoi[MASK_SYMBOL]

