#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import json
import os
import tqdm
import time
from run_pretrain import BehaviorsBERTTokenizer, max_seq_len, construct_pos_mask
from corpus_vocab import vocab_file, SOS_SYMBOL, EOS_SYMBOL, PAD_SYMBOL
from dataset import str2timestamp


class BERTFullInference(nn.Module):
    def __init__(self, transformer, sequence_summary_type='last'):
        super(BERTFullInference, self).__init__()
        self.transformer = transformer
        self.sequence_summary_type = sequence_summary_type

    def forward(self, token_seq, time_seq, pos_mask):
        if len(pos_mask.shape) == 4:
            pos_mask = pos_mask.squeeze(0)

        outputs = self.transformer(token_seq, time_seq, pos_mask)
        if self.sequence_summary_type == 'first':
            return outputs[: 0, :]
        elif self.sequence_summary_type == 'last':
            return outputs[: -1, :]
        elif self.sequence_summary_type == 'max_pooling':
            return torch.max(outputs, dim=1)[0]
        elif self.sequence_summary_type == 'mean_pooling':
            return torch.mean(outputs, dim=1)
        else:
            raise NotImplementedError("sequence_summary_type = [first, last, max_pooling, mean_pooling]")


def get_churn_prediction_by_days_embeddings(input_seq_dir, output_embds_file):
    files = [os.path.join(input_seq_dir, e) for e in os.listdir(input_seq_dir)]
    n_batches = (len(files) + batch_size - 1) // batch_size

    fout = open(output_embds_file, 'w')

    for i in tqdm.tqdm(range(n_batches), desc=output_embds_file):
        batch_files = files[i * batch_size: (i+1) * batch_size]
        batch_token_seq, batch_time_seq = [], []
        batch_indices = []
        for _file in batch_files:
            with open(_file, 'r', encoding='utf-8') as fin:
                # 2019-08-01:400347#0#2019-08-01-00:00:00 400616#0#2019-08-01-00:00:00 400177#0#2019-08-01-00:00:01
                for l in fin:
                    day = l[:l.index(':')]
                    events = l[l.index(':')+1:].strip().split(' ')
                    events = events[-(max_seq_len - 2):]

                    token_seq = ["#".join(e.split('#')[:2]) for e in events]
                    time_seq = [str2timestamp(e.split('#')[-1]) for e in events]
                    min_t = min(time_seq)
                    time_seq = [t - min_t + 1 for t in time_seq]

                    token_seq = [SOS_SYMBOL] + token_seq + [EOS_SYMBOL]
                    time_seq = [0.0] + time_seq + [0.0]
                    padding = [PAD_SYMBOL for _ in range(max_seq_len - len(token_seq))]
                    token_seq = padding + token_seq
                    time_seq = [0] * len(padding) + time_seq

                    token_seq = tokenizer.convert_tokens_to_ids(token_seq)

                    batch_token_seq.append(token_seq)
                    batch_time_seq.append(time_seq)
                    batch_indices.append("%s,%s" % (os.path.basename(_file), day))

        with torch.no_grad():
            n_batches_batches = (len(batch_token_seq) + batch_size - 1) // batch_size
            for j in range(n_batches_batches):
                _indices = batch_indices[j * batch_size: (j+1) * batch_size]
                _input_token_seq = batch_token_seq[j * batch_size: (j+1) * batch_size]
                _input_time_seq = batch_time_seq[j * batch_size: (j+1) * batch_size]
                outputs = model(torch.tensor(_input_token_seq).to(device),
                                torch.tensor(_input_time_seq).to(device),
                                pos_mask)
                outputs = outputs.tolist()
                # role_id, ds, *portrait_features = l.strip().split(',')
                for idx, output in zip(_indices, outputs):
                    fout.write("%s,%s\n" % (idx, ','.join(map(str, output))))
    fout.close()


def get_map_preload_embeddings(input_seq_dirs, output_embds_file):
    files = []
    for _seq_dir in input_seq_dirs:
        if not os.path.isdir(_seq_dir):
            continue
        day = os.path.basename(_seq_dir).split('.')[-1]
        for e in os.listdir(_seq_dir):
            files.append((os.path.join(_seq_dir, e), day))

    n_batches = (len(files) + batch_size - 1) // batch_size

    fout = open(output_embds_file, 'w')
    for i in tqdm.tqdm(range(n_batches), desc=output_embds_file):
        batch_files = files[i * batch_size: (i + 1) * batch_size]
        batch_token_seq, batch_time_seq = [], []
        batch_indices = []
        for _file in batch_files:
            with open(_file[0], 'r', encoding='utf-8') as fin:
                events = json.load(fin)
                events = events[-(max_seq_len - 2):]

                time_seq = [str2timestamp(e.split('#')[0].replace(' ', '-')) for e in events]
                token_seq = ['#'.join(e.split('#')[-2:]) for e in events]
                min_t = min(time_seq)
                time_seq = [t - min_t + 1 for t in time_seq]

                token_seq = [SOS_SYMBOL] + token_seq + [EOS_SYMBOL]
                time_seq = [0.0] + time_seq + [0.0]
                padding = [PAD_SYMBOL for _ in range(max_seq_len - len(token_seq))]
                token_seq = padding + token_seq
                time_seq = [0] * len(padding) + time_seq

                token_seq = tokenizer.convert_tokens_to_ids(token_seq)

                batch_token_seq.append(token_seq)
                batch_time_seq.append(time_seq)
                batch_indices.append("{},{}".format(os.path.basename(_file[0]), _file[1]))

        with torch.no_grad():
            outputs = model(torch.tensor(batch_token_seq).to(device),
                            torch.tensor(batch_time_seq).to(device),
                            pos_mask)
            outputs = outputs.tolist()
            for idx, output in zip(batch_indices, outputs):
                fout.write("%s,%s\n" % (idx, ','.join(map(str, output))))
    fout.close()


def get_bot_detection_embeddings(input_seq_dir, output_embds_file):
    files = [os.path.join(input_seq_dir, e) for e in os.listdir(input_seq_dir)]
    n_batches = (len(files) + batch_size - 1) // batch_size

    fout = open(output_embds_file, 'w')

    for i in tqdm.tqdm(range(n_batches), desc=output_embds_file):
        batch_files = files[i * batch_size: (i+1) * batch_size]
        batch_token_seq, batch_time_seq = [], []
        batch_indices = []
        for _file in batch_files:
            with open(_file, 'r', encoding='utf-8') as fin:
                events = json.load(fin)
                events = events[-(max_seq_len - 2):]

                time_seq = [str2timestamp(e["timestamp"].replace(' ', '-')) for e in events]
                token_seq = ['%s#%s' % (e['log_id'], e.get('design_id', 0)) for e in events]
                min_t = min(time_seq)
                time_seq = [t - min_t + 1 for t in time_seq]

                token_seq = [SOS_SYMBOL] + token_seq + [EOS_SYMBOL]
                time_seq = [0.0] + time_seq + [0.0]
                padding = [PAD_SYMBOL for _ in range(max_seq_len - len(token_seq))]
                token_seq = padding + token_seq
                time_seq = [0] * len(padding) + time_seq

                token_seq = tokenizer.convert_tokens_to_ids(token_seq)

                batch_token_seq.append(token_seq)
                batch_time_seq.append(time_seq)
                batch_indices.append(os.path.basename(_file).replace(':', '-'))
        with torch.no_grad():
            outputs = model(torch.tensor(batch_token_seq).to(device),
                            torch.tensor(batch_time_seq).to(device),
                            pos_mask)
            outputs = outputs.tolist()
            for idx, output in zip(batch_indices, outputs):
                fout.write("%s,%s\n" % (idx, ','.join(map(str, output))))

    fout.close()


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2
    model_tag = 'bert_full'
    model_path = os.path.join("bert-full", sys.argv[1])
    assert os.path.exists(model_path), "{} not exists!!".format(model_path)

    tokenizer = BehaviorsBERTTokenizer(vocab_file=vocab_file)
    print("Vocab size:", tokenizer.vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu_or_cpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = 8 * n_gpu_or_cpu

    transformer = torch.load(model_path, map_location=device)
    # get_words_embeddings(); exit(0)

    model = BERTFullInference(transformer=transformer, sequence_summary_type='max_pooling').to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    pos_mask = construct_pos_mask().to(device)
    if n_gpu_or_cpu == 1:
        pass
    else:
        pos_mask = pos_mask.unsqueeze(0).repeat([n_gpu_or_cpu, 1, 1, 1])

    print("Inference start....")
    print("  device:{}".format(device))
    print("  batch_size:{}, n_gpu_or_cpu:{} per gpu size:{}".format(batch_size, n_gpu_or_cpu, batch_size / n_gpu_or_cpu))

    t0 = time.time()

    churn_data_by_days_dir = '/GameBERT/dataset/churn-prediction'
    output_dir = '/GameBERT/dataset/churn-prediction'
    churn_servers = ['242.pos', '289.pos', '242.neg', '289.neg']
    for s in churn_servers:
        input_seq_dir = os.path.join(churn_data_by_days_dir, s)
        output_embds_file = os.path.join(output_dir, '{}.behaviors_vectors_{}'.format(s, model_tag))
        get_churn_prediction_by_days_embeddings(input_seq_dir, output_embds_file)

    t1 = time.time()
    print("  churn prediction cost:%.4fs" % (t1 - t0))

    base_dir = '/GameBERT/dataset/map-preload-prediction'
    output_dir = '/GameBERT/dataset/map-preload-prediction'
    servers = [242, 204]
    for s in servers:
        input_seq_dirs = sorted([os.path.join(base_dir, e) for e in os.listdir(base_dir) if str(s) in e])
        output_embds_file = os.path.join(output_dir, "{}.{}.behaviors_vectors".format(s, model_tag))
        get_map_preload_embeddings(input_seq_dirs, output_embds_file)

    t2 = time.time()
    print("  map preload cost:%.4fs" % (t2 - t1))

    seq_dir = '/GameBERT/dataset/bot-detection'
    output_dir = '/GameBERT/dataset/bot-detection'
    choices = ['pos', 'neg']
    for c in choices:
        input_seq_dir = os.path.join(seq_dir, c)
        output_embds_file = os.path.join(output_dir, '{}.{}'.format(c, model_tag))
        get_bot_detection_embeddings(input_seq_dir, output_embds_file)

    t3 = time.time()
    print("  bot detection cost:%.4fs" % (t3 - t2))
    print("Total cost:%.4fs" % (t3 - t0))
