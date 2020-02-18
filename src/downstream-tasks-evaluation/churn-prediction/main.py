#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import tensorflow as tf
from data_preprocess import seed, train_dir, test_dir

np.random.seed(seed)
try:
    tf.compat.v1.set_random_seed(seed)
except Exception as e:
    print("Warning:{}".format(e))

from model import train_model


def load_data(server, model_tag):
    with open(os.path.join(train_dir, str(server), 'scaled_X1_{}.pkl'.format(model_tag)), 'rb') as f:
        X1_train = pickle.load(f)
    with open(os.path.join(train_dir, str(server), 'y.pkl'), 'rb') as f:
        y_train = np.asarray(pickle.load(f))

    with open(os.path.join(test_dir, str(server), 'scaled_X1_{}.pkl'.format(model_tag)), 'rb') as f:
        X1_test = pickle.load(f)
    with open(os.path.join(test_dir, str(server), 'y.pkl'), 'rb') as f:
        y_test = np.asarray(pickle.load(f))

    n_train_samples = X1_train.shape[0]
    n_test_samples = X1_test.shape[0]

    train_indices = np.arange(n_train_samples)
    np.random.shuffle(train_indices)
    X1_train, y_train = X1_train[train_indices], y_train[train_indices]

    test_indices = np.arange(n_test_samples)
    np.random.shuffle(test_indices)
    X1_test, y_test = X1_test[test_indices], y_test[test_indices]

    return X1_train, X1_test, y_train, y_test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Consumer")
    parser.add_argument('--server', type=str, required=True)
    parser.add_argument('--model_tag', type=str, required=True)
    args = parser.parse_args()

    server = args.server
    model_tag = args.model_tag

    X1_train, X1_test, y_train, y_test = load_data(server, model_tag)

    assert X1_train.shape[0] == y_train.shape[0], "X1_train.shape={}, y_train.shape={}".format(X1_train.shape,
                                                                                               y_train.shape)

    assert X1_test.shape[0] == y_test.shape[0], "X1_test.shape={}, y_test.shape={}".format(X1_test.shape,
                                                                                           y_test.shape)

    assert X1_train.shape[-1] == X1_test.shape[-1], "X1_train.shape={}, X1_test.shape={}".format(X1_train.shape,
                                                                                                 X1_test.shape)
    assert X1_train.shape[1] == X1_test.shape[1] == 30, "X1_train.shape={}, X1_test.shape={}".format(X1_train.shape,
                                                                                                     X1_test.shape)
    feature_dims = X1_test.shape[-1]

    print("feature_dims=", feature_dims)

    lr = 0.0001
    epochs = 100
    num_gpus = 1
    batch_size = 128 * num_gpus

    train_model(X1_train, X1_test, y_train, y_test,
                lr, epochs, num_gpus, batch_size, feature_dims,
                use_multi_gpu=num_gpus > 1)
