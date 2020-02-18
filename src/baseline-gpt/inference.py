#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import tqdm
import json
import time
from modeling_utils import SequenceSummary
from configuration_openai import OpenAIGPTConfig
from run_openai_gpt import BehaviorsGPTTokenizer
from corpus_vocab import vocab_file, SOS_SYMBOL, EOS_SYMBOL, PAD_SYMBOL


class GPTInference(nn.Module):
    def __init__(self, config, transformer):
        super(GPTInference, self).__init__()
        self.transformer = transformer
        self.sequencesummary = SequenceSummary(config)

    def forward(self, input_ids):
        return self.sequencesummary(self.transformer(input_ids))


def get_words_embeddings():
    tokenEmb = transformer.state_dict()['tokens_embed.weight']
    print('tokenEmb.shape', tokenEmb.shape)
    tokenEmb = tokenEmb.tolist()
    itos = {i:s for s, i in tokenizer.stoi.items()}
    print(len(itos))

    with open(model_path + '.tokensEmbedding', 'w') as fout:
        for i, emb in enumerate(tokenEmb):
            fout.write("%s:%s\n" % (itos[i], ",".join(map(str, emb))))


def get_churn_prediction_by_days_embeddings(input_seq_dir, output_embds_file):
    files = [os.path.join(input_seq_dir, e) for e in os.listdir(input_seq_dir)]
    n_batches = (len(files) + batch_size - 1) // batch_size

    fout = open(output_embds_file, 'w')

    for i in tqdm.tqdm(range(n_batches), desc=output_embds_file):
        batch_files = files[i * batch_size: (i+1) * batch_size]
        batch_input_ids = []
        batch_indices = []
        for _file in batch_files:
            with open(_file, 'r', encoding='utf-8') as fin:
                for l in fin:
                    day = l[:l.index(':')]
                    events = l[l.index(':')+1:].strip().split(' ')
                    events = events[-(config.max_position_embeddings - 2):]

                    events = ["#".join(e.split('#')[:2]) for e in events]

                    events = [SOS_SYMBOL] + events + [EOS_SYMBOL]
                    padding = [PAD_SYMBOL for _ in range(config.max_position_embeddings - len(events))]
                    events.extend(padding)
                    ids = tokenizer.convert_tokens_to_ids(events)

                    batch_input_ids.append(ids)
                    batch_indices.append("%s,%s" % (os.path.basename(_file), day))

        with torch.no_grad():
            n_batches_batches = (len(batch_input_ids) + batch_size - 1) // batch_size
            for j in range(n_batches_batches):
                _input_ids = batch_input_ids[j * batch_size: (j+1) * batch_size]
                _indices = batch_indices[j * batch_size: (j+1) * batch_size]
                outputs = model(torch.tensor(_input_ids).to(device))
                outputs = outputs.tolist()
                for idx, output in zip(_indices, outputs):
                    fout.write("%s,%s\n" % (idx, ','.join(map(str, output))))


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
        batch_input_ids = []
        batch_indices = []
        for _file in batch_files:
            with open(_file[0], 'r', encoding='utf-8') as fin:
                events = json.load(fin)
                events = ['#'.join(e.split('#')[-2:]) for e in events]

                events = events[-(config.max_position_embeddings - 2):]
                events = [SOS_SYMBOL] + events + [EOS_SYMBOL]
                padding = [PAD_SYMBOL for _ in range(config.max_position_embeddings - len(events))]
                events.extend(padding)

                batch_input_ids.append(tokenizer.convert_tokens_to_ids(events))
                batch_indices.append("{},{}".format(os.path.basename(_file[0]), _file[1]))

        with torch.no_grad():
            outputs = model(torch.tensor(batch_input_ids).to(device))
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
        batch_input_ids = []
        batch_indices = []
        for _file in batch_files:
            with open(_file, 'r', encoding='utf-8') as fin:
                events = json.load(fin)
                events = ['%s#%s' % (e['log_id'], e.get('design_id', 0)) for e in events]

                events = events[-(config.max_position_embeddings - 2):]
                events = [SOS_SYMBOL] + events + [EOS_SYMBOL]
                padding = [PAD_SYMBOL for _ in range(config.max_position_embeddings - len(events))]
                events.extend(padding)

                batch_input_ids.append(tokenizer.convert_tokens_to_ids(events))
                batch_indices.append(os.path.basename(_file).replace(':', '-'))
        with torch.no_grad():
            outputs = model(torch.tensor(batch_input_ids).to(device))
            outputs = outputs.tolist()
            for idx, output in zip(batch_indices, outputs):
                fout.write("%s,%s\n" % (idx, ','.join(map(str, output))))
    fout.close()


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2
    model_path = os.path.join('gpt', sys.argv[1])

    model_tag = 'gpt512'
    tokenizer = BehaviorsGPTTokenizer(vocab_file=vocab_file, freq_thres=1000)
    print("Vocab size:", tokenizer.vocab_size)

    config = OpenAIGPTConfig(
        summary_type='cls_index',
        summary_use_proj=False,
        summary_activation=None,
        summary_proj_to_labels=False,
        summary_first_dropout=0.1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpu_or_cpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = 128 * n_gpu_or_cpu

    transformer = torch.load(model_path, map_location=device)
    # get_words_embeddings(); exit(0)

    model = GPTInference(config=config, transformer=transformer).to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

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
