#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import BERTLMLSS, BERTLSS
from optim_schedule import ScheduledOptim
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter



class BasicTrainer:
    def __init__(self, bert: BERTLSS, epochs: int,
                 tensorboard_log_dir: str,
                 output_path: str,
                 train_dataloader: DataLoader,
                 with_cuda: bool, log_freq: int, save_steps: int):
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.train_data = train_dataloader
        self.n_batches = len(train_dataloader)
        print("Total epochs:%s, Batches per epoch:%s" % (epochs, self.n_batches))
        # This BERT model will be saved every epoch
        self.bert = bert

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        # ignore_index – Specifies a target value that is ignored and does not contribute to the input gradient.
        # When size_average is True, the loss is averaged over non-ignored targets.
        # 所以，类别0没有被计入loss计算，也就达到了mask的目的。
        self.mask_lm_criterion = nn.NLLLoss(ignore_index=0)
        # self.mask_time_prediction_criterion = MaskTimePredictionLoss()

        self.log_freq = log_freq

        self.model = None
        self.tensorborad_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        self.epochs = epochs
        self.save_steps = save_steps
        self.output_path = output_path

    def train(self, pos_mask):
        self.model.train()
        for epoch in range(self.epochs):
            self.iteration(epoch, self.train_data, pos_mask)
        self.tensorborad_writer.close()

    def iteration(self, epoch, data_loader, pos_mask):
        raise NotImplementedError

    def save(self, epoch, iteration):
        output_path = self.output_path + ".ep%d.iter%d" % (epoch, iteration)
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Iter:%d Model Saved on:" % (epoch, iteration), output_path)


class BERTTrainerLSS(BasicTrainer):
    def __init__(self, bert: BERTLSS, vocab_size: int, epochs: int,
                 tensorboard_log_dir: str, output_path: str,
                 train_dataloader: DataLoader,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, log_freq: int = 10, save_steps: int = -1):

        super(BERTTrainerLSS, self).__init__(bert=bert, epochs=epochs,
                                             tensorboard_log_dir=tensorboard_log_dir,
                                             output_path=output_path,
                                             train_dataloader=train_dataloader,
                                             with_cuda=with_cuda, log_freq=log_freq, save_steps=save_steps)

        self.model = BERTLMLSS(bert, vocab_size).to(self.device)

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def iteration(self, epoch, data_loader, pos_mask):
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP:%d" % epoch,
                              total=self.n_batches,
                              bar_format="{l_bar}{r_bar}",
                              disable=False)

        avg_loss = 0.0
        for i, data in data_iter:
            global_step = epoch * self.n_batches + i + 1

            data = {key: value.to(self.device) for key, value in data.items()}

            mask_lm_output = self.model.forward(data["token_input"], pos_mask)

            mask_lm_loss = self.mask_lm_criterion(mask_lm_output.transpose(1, 2), data["token_label"])
            del data

            loss = mask_lm_loss

            self.optim_schedule.zero_grad()
            loss.backward()
            self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            self.tensorborad_writer.add_scalar("Masked_language_model loss", mask_lm_loss.item(), global_step)
            self.tensorborad_writer.add_scalar("Average loss in epoch", avg_loss / (i + 1), global_step)

            post_fix = {
                "epoch": epoch,
                "iter": i+1,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if (i+1) % self.log_freq == 0:
                data_iter.write(str(post_fix))

            if self.save_steps > 0 and ((i + 1) % self.save_steps == 0 or (i + 1) == self.n_batches):
                self.save(epoch, i + 1)
            # input("One batch stops\n"
            #       "memory_allocated={} max_memory_allocated={}\n"
            #       "memory_cached={} max_memory_cached={}\n".format(
            #     torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            #     torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
            #     torch.cuda.memory_cached() / 1024 / 1024 / 1024,
            #     torch.cuda.max_memory_cached() / 1024 / 1024 / 1024
            # ))
            #
            # torch.cuda.empty_cache()
            # input("after cuda.empty_cache\n"
            #       "memory_allocated={} max_memory_allocated={}\n"
            #       "memory_cached={} max_memory_cached={}\n".format(
            #     torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
            #     torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
            #     torch.cuda.memory_cached() / 1024 / 1024 / 1024,
            #     torch.cuda.max_memory_cached() / 1024 / 1024 / 1024
            # ))

