#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import gc
import tqdm
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import precision_score, recall_score

base_dir = '/GameBERT/dataset/bot-detection'
pos_feature_file = os.path.join(base_dir, 'pos.{}')
neg_feature_file = os.path.join(base_dir, 'neg.{}')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


batch_size = 128
epochs = 50
dropout_rate = 0.2
dense_size_1 = 256
regular_lambda = 0.002
dense_size_2 = 64

input_dim = 256


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


def build_model(n_gpu):
    with tf.device('/cpu:0'):
        # Inputs
        model = Sequential()
        model.add(Dense(dense_size_1, input_dim=input_dim, activation='relu',
                        kernel_regularizer=regularizers.l1(regular_lambda)))
        model.add(Dense(dense_size_2, activation='relu',
                        kernel_regularizer=regularizers.l1(regular_lambda)))
        model.add(Dense(1, activation='sigmoid'))
    if n_gpu > 1:
        model = multi_gpu_model(model, gpus=n_gpu)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_score])
    model.summary()

    return model


def load_dataset(model_tag):
    def __load_features(feature_file: str, _dir: str):
        samples = os.listdir(_dir)
        samples = [e.replace(':', '-') for e in samples]

        features_d = {}
        with open(feature_file, 'r') as fin:
            for l in fin:
                sample, *feature = l.strip().split(',')
                features_d[sample] = feature

        features = []
        for sample in tqdm.tqdm(samples):
            features.append(features_d[sample])
        return features

    pos_train_features = np.asarray(__load_features(pos_feature_file.format(model_tag), train_dir))
    neg_train_features = np.asarray(__load_features(neg_feature_file.format(model_tag), train_dir))
    pos_test_features = np.asarray(__load_features(pos_feature_file.format(model_tag), test_dir))
    neg_test_features = np.asarray(__load_features(neg_feature_file.format(model_tag), test_dir))

    print(pos_train_features.shape)
    print(neg_train_features.shape)
    print(pos_test_features.shape)
    print(neg_test_features.shape)

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
    parser.add_argument('--model_tag', type=str, required=True)
    args = parser.parse_args()

    n_gpu = int(args.n_gpu)
    model_tag = args.model_tag

    t1 = time.time()
    train_features, train_y, test_features, test_y = load_dataset(model_tag)
    t2 = time.time()
    print("Load dataset cost: [%.4fs]" % (t2 - t1))
    print(type(train_features), train_features.shape)
    print(type(train_y), train_y.shape)
    print(type(test_features), test_features.shape)
    print(type(test_y), test_y.shape)

    model = build_model(n_gpu)
    model.fit(x=train_features,
              y=train_y,
              batch_size=batch_size * n_gpu, epochs=epochs,
              callbacks=[Metrics()],
              verbose=1,
              validation_data=(test_features, test_y),
              shuffle=True)

