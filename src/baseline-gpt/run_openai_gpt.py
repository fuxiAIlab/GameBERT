#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import copy
import random
import math
import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from corpus_vocab import specials, UNK_SYMBOL, vocab_file, PAD_SYMBOL, SOS_SYMBOL, EOS_SYMBOL
from modeling_openai import OpenAIGPTLMHeadModel
from configuration_openai import OpenAIGPTConfig
import argparse
rdn_seed = 123456789

parser = argparse.ArgumentParser()
parser.add_argument('--max_seq_len', type=int, default=512)
parser.add_argument('--per_gpu_batch_size', type=int, default=16)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--adam_epsilon', type=float, default=1e-8)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--num_train_epochs', type=int, default=500)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=16)
parser.add_argument('--n_embd', type=int, default=256)
parser.add_argument('--save_steps', type=int, default=-1)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

all_dataset_dir = '/GameBERT/dataset/corpus'
max_seq_len = args.max_seq_len
n_positions = max_seq_len
per_gpu_batch_size = args.per_gpu_batch_size
weight_decay = args.weight_decay
learning_rate = args.learning_rate
adam_epsilon = args.adam_epsilon
warmup_steps = args.warmup_steps
gradient_accumulation_steps = args.gradient_accumulation_steps
num_train_epochs = args.num_train_epochs
max_grad_norm = args.max_grad_norm
n_layer = args.n_layer
n_head = args.n_head
n_embd = args.n_embd
save_steps = args.save_steps

verbose = args.verbose

corpus_lines = 24381321

params_str = "_{}_{}_{}_{}_{}".format(max_seq_len, n_embd, n_layer, n_head, rdn_seed)

output_dir = 'gpt'
model_out_dir = output_dir + "/model" + params_str
tensorboard_log_dir = output_dir + '/tensorboard' + params_str


class BehaviorsGPTTokenizer:
    def __init__(self, vocab_file):
        itos = copy.deepcopy(specials)
        with open(vocab_file, 'r', encoding='utf-8') as fin:
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


class BehaviorsGPTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, on_memory=False, corpus_lines=None):
        self.tokenizer = tokenizer
        if on_memory:
            raise NotImplementedError

        self.corpus_lines = 0
        self.blocks = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
        self.n_blocks = len(self.blocks)
        if corpus_lines is None:
            for block in tqdm.tqdm(self.blocks, desc="Total %s blocks" % self.n_blocks, disable=False):
                with open(block, 'r') as fin:
                    for _ in tqdm.tqdm(fin, desc="load block %s" % block, disable=True):
                        self.corpus_lines += 1
        else:
            self.corpus_lines = corpus_lines
        self.block_idx = 0
        self.block_file_handlers = []
        for block in self.blocks:
            self.block_file_handlers.append([open(block, 'r'), block])

    def __len__(self):
        return self.corpus_lines

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
                line = line.split(':')[1].split(' ')
                if len(line) > 1:
                    return line

    def __getitem__(self, item):
        line = self.__get_one_line()

        length = len(line)
        start_idx = random.randint(0, max(0, length - (max_seq_len - 2)))
        ids = self.tokenizer.convert_tokens_to_ids(line[start_idx: start_idx + max_seq_len - 2])
        ids = [self.tokenizer.convert_tokens_to_ids(SOS_SYMBOL)] + ids + [self.tokenizer.convert_tokens_to_ids(EOS_SYMBOL)]

        padding = [self.tokenizer.convert_tokens_to_ids(PAD_SYMBOL) for _ in range(max_seq_len - len(ids))]
        ids.extend(padding)  # post-padding

        return torch.tensor(ids, dtype=torch.long)


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.

    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1]  < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        correct_bias=correct_bias)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)

        return loss


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


def main():
    random.seed(rdn_seed)
    np.random.seed(rdn_seed)
    torch.manual_seed(rdn_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {}, n_gpu: {}".format(device, n_gpu))
    if device == "cuda": torch.cuda.manual_seed_all(rdn_seed)

    tokenizer = BehaviorsGPTTokenizer(vocab_file=vocab_file, freq_thres=1000)
    print("Vocab size:", tokenizer.vocab_size)

    config = OpenAIGPTConfig(vocab_size=tokenizer.vocab_size,
                             n_layer=n_layer,
                             n_head=n_head,
                             n_positions=n_positions,
                             n_ctx=n_positions,
                             n_embd=n_embd,
                             verbose=verbose)
    model = OpenAIGPTLMHeadModel(config)
    print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))
    model = model.to(device)
    if n_gpu > 1:
        print("Using {} gpus for paralleling training".format(n_gpu))
        model = nn.DataParallel(model)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    dataset = BehaviorsGPTDataset(all_dataset_dir, tokenizer, corpus_lines=corpus_lines)
    print('Dataset length', len(dataset))
    train_batch_size = per_gpu_batch_size * (1 if n_gpu == 0 else n_gpu) * gradient_accumulation_steps
    print("Train batch size:", train_batch_size, 'Per gpu batch size:', per_gpu_batch_size)
    dataloader = DataLoader(dataset, batch_size=train_batch_size)
    t_total = len(dataloader) // gradient_accumulation_steps * num_train_epochs
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    tensorborad_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    model.train()
    # if verbose: input("Before training CUDA memory allocated=%s, Press Enter to continue..." % torch.cuda.memory_allocated())
    for epoch in range(num_train_epochs):
        data_iter = tqdm.tqdm(dataloader, desc="EP_%s" % epoch, total=len(dataloader), bar_format="{l_bar}{r_bar}", disable=False)
        n_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            inputs, labels = batch, batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            if verbose: print("transfer to %s done." % device)

            outputs = model(inputs, labels=labels)

            loss = outputs[0]
            if n_gpu > 1: loss = loss.mean()
            if gradient_accumulation_steps > 1: loss /= gradient_accumulation_steps

            loss.backward()
            batch_loss = loss.item()
            tr_loss += batch_loss

            if (i+1) % gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                avg_loss = tr_loss / global_step
                tensorborad_writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
                tensorborad_writer.add_scalar("Loss", batch_loss, global_step)
                tensorborad_writer.add_scalar("Average loss", avg_loss, global_step)
                print_msg = {
                    "Epoch": epoch,
                    "iter": i + 1,
                    "global_step": global_step,
                    "loss": batch_loss,
                    "avg_loss": avg_loss
                }
                if (i+1) % 100 == 0:
                    data_iter.write(str(print_msg))

                if save_steps > 0 and ((i+1) % save_steps == 0 or (i+1) == n_batches):
                    if isinstance(model, nn.DataParallel):
                        torch.save(model.module.transformer.cpu(),
                                   os.path.join(model_out_dir, ".ep%s.iter%s" % (epoch, i + 1)))
                        model.module.transformer.to(device)
                    elif isinstance(model, OpenAIGPTLMHeadModel):
                        torch.save(model.transformer.cpu(),
                                   os.path.join(model_out_dir, ".ep%s.iter%s" % (epoch, i + 1)))
                        model.transformer.to(device)

    print("Training is Done.")
    tensorborad_writer.close()


if __name__ == "__main__":
    main()
