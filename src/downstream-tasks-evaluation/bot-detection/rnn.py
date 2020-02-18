#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import gc
import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import precision_score, recall_score


base_dir = '/GameBERT/dataset/bot-detection'
train_pos_dir = os.path.join(base_dir, 'train', 'pos')
train_neg_dir = os.path.join(base_dir, 'train', 'neg')
test_pos_dir = os.path.join(base_dir, 'test', 'pos')
test_neg_dir = os.path.join(base_dir, 'test', 'neg')


batch_size = 128
embed_size = 128
epochs = 50
dropout_rate = 0.2
max_len = 1024

vocab_file = os.path.join(base_dir, 'vocab.logdesignid')


def f1_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = c1 / (c2 + K.epsilon())
    recall = c1 / (c3 + K.epsilon())
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_f1 = 2 * _val_precision * _val_recall / (_val_precision + _val_recall)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('\nEpoch:%s- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f\n' % (epoch, _val_f1, _val_precision, _val_recall))
        return


def build_model(n_gpu, vocab):
    with tf.device('/gpu'):
        # Inputs    
        model = Sequential()

        model.add(Embedding(vocab.vocab_size, embed_size))
        model.add(LSTM(embed_size, dropout=dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

    if n_gpu > 1:
        model = multi_gpu_model(model, gpus=n_gpu)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    model.summary()

    return model


class Vocab:
    UNK_TOKEN = 'unknown'

    def __init__(self, vocab_file):
        tokens = [Vocab.UNK_TOKEN]
        with open(vocab_file, 'r') as fin:
            for l in fin:
                tokens.append(l.strip())
        self.data = {}
        for idx, token in enumerate(tokens):
            self.data[token] = idx
        self.rdata = {v:k for k, v in self.data.items()}

    @property
    def vocab_size(self):
        return len(self.data)

    def token2idx(self, token):
        return self.data.get(token, 0)

    def idx2token(self, idx):
        return self.rdata[idx]


def load_dataset(vocab):
    def __load_features(_dir: str, jsonfilepath: str, vocab: Vocab):
        samples = os.listdir(_dir)
        features = []
        for sample in tqdm.tqdm(samples, desc=jsonfilepath):
            feature = []
            with open(os.path.join(_dir, sample), 'r', encoding='utf-8') as fin:
                events = json.load(fin)
                for event in events:
                    logdesignid = "%s#%s" % (event["log_id"], event.get("design_id", 0))
                    feature.append(vocab.token2idx(logdesignid))
            padding = [0 for _ in range(max_len - len(feature))]
            feature = padding + feature
            features.append(feature[-max_len:])
        return features
    pos_train_features = np.asarray(__load_features(train_pos_dir, vocab))
    neg_train_features = np.asarray(__load_features(train_neg_dir, vocab))
    pos_test_features = np.asarray(__load_features(test_pos_dir, vocab))
    neg_test_features = np.asarray(__load_features(test_neg_dir, vocab))

    train_y = np.asarray([1] * pos_train_features.shape[0] + [0] * neg_train_features.shape[0])
    test_y = np.asarray([1] * pos_test_features.shape[0] + [0] * neg_test_features.shape[0])
    train_features = np.concatenate((pos_train_features, neg_train_features), axis=0)
    test_features = np.concatenate((pos_test_features, neg_test_features), axis=0)
    del pos_train_features, neg_train_features, pos_test_features, neg_test_features
    gc.collect()

    train_indices = np.random.permutation(len(train_y))
    test_indices = np.random.permutation(len(test_y))
    train_features, train_y = train_features[train_indices], train_y[train_indices]
    test_features, test_y = test_features[test_indices], test_y[test_indices]

    return train_features, train_y, test_features, test_y


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Consumer")
    parser.add_argument('--n_gpu', type=str, required=True)
    args = parser.parse_args()

    n_gpu = int(args.n_gpu)

    t1 = time.time()
    vocab = Vocab(vocab_file)
    train_features, train_y, test_features, test_y = load_dataset(vocab)
    t2 = time.time()
    print("Load dataset cost: [%.4fs]" % (t2 - t1))
    print(type(train_features), train_features.shape)
    print(type(train_y), train_y.shape)
    print(type(test_features), test_features.shape)
    print(type(test_y), test_y.shape)

    model = build_model(n_gpu, vocab)
    model.fit(x=train_features,
              y=train_y,
              batch_size=batch_size * n_gpu, epochs=epochs,
              callbacks=[Metrics()],
              verbose=1,
              validation_data=(test_features, test_y),
              shuffle=True)

