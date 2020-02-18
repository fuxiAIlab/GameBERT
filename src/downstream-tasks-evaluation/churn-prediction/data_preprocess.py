#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gc
import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import tqdm

import datetime


def ds_range(a, b):
    day_format = '%Y-%m-%d'
    da = datetime.datetime.strptime(a, day_format)
    i = 0
    tmp = (da + datetime.timedelta(days=i)).strftime(day_format)
    while tmp <= b:
        yield tmp
        i += 1
        tmp = (da + datetime.timedelta(days=i)).strftime(day_format)


base_dir = '/GameBERT/dataset/churn-prediction'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
seed = 1024
np.random.seed(seed)

ds_ranges = list(ds_range('2019-08-01', '2019-08-30'))


def split_to_train_test(server):
    pos_dir = os.path.join(base_dir, '{}.pos'.format(server))
    neg_dir = os.path.join(base_dir, '{}.neg'.format(server))
    pos_role_ids, neg_role_ids = list(map(int, os.listdir(pos_dir))), list(map(int, os.listdir(neg_dir)))
    pos_role_ids_train, pos_role_ids_test = train_test_split(pos_role_ids, test_size=0.2, random_state=seed)
    neg_role_ids_train, neg_role_ids_test = train_test_split(neg_role_ids, test_size=0.2, random_state=seed)

    with open(os.path.join(train_dir, str(server), 'pos_role_ids_train.json'), 'w', encoding='utf-8') as fout:
        json.dump(pos_role_ids_train, fout, indent=4)
    with open(os.path.join(test_dir, str(server), 'pos_role_ids_test.json'), 'w', encoding='utf-8') as fout:
        json.dump(pos_role_ids_test, fout, indent=4)
    with open(os.path.join(train_dir, str(server), 'neg_role_ids_train.json'), 'w', encoding='utf-8') as fout:
        json.dump(neg_role_ids_train, fout, indent=4)
    with open(os.path.join(test_dir, str(server), 'neg_role_ids_test.json'), 'w', encoding='utf-8') as fout:
        json.dump(neg_role_ids_test, fout, indent=4)


def get_train_dataset(server, behaviors_portrait_dict, origin_portrait_dict, model_tag):
    with open(os.path.join(train_dir, str(server), 'pos_role_ids_train.json'), 'r', encoding='utf-8') as fin:
        pos_role_ids = json.load(fin)
    with open(os.path.join(train_dir, str(server), 'neg_role_ids_train.json'), 'r', encoding='utf-8') as fin:
        neg_role_ids = json.load(fin)

    role_ids = pos_role_ids + neg_role_ids
    y = [1] * len(pos_role_ids) + [0] * len(neg_role_ids)

    role_origin_portraits = []
    for role_id in tqdm.tqdm(role_ids, desc="role_origin_portraits"):
        if role_id in origin_portrait_dict:
            data = origin_portrait_dict[role_id]
            vecs = []
            for ds in ds_ranges:
                vec = data.get(ds, [0] * origin_portrait_dim)
                vecs.append(vec)
            role_origin_portraits.append(vecs)
        else:
            role_origin_portraits.append([[0] * origin_portrait_dim for _ in ds_ranges])
    role_origin_portraits = np.asarray(role_origin_portraits)

    if behaviors_portrait_dict == {}:
        role_behaviors_portraits = np.zeros(shape=(role_origin_portraits.shape[0],
                                                   role_origin_portraits.shape[1],
                                                   256), dtype=np.float32)
    else:
        role_behaviors_portraits = []
        for role_id in tqdm.tqdm(role_ids, desc="role_behaviors_portraits"):
            if role_id in behaviors_portrait_dict:
                data = behaviors_portrait_dict[role_id]
                vecs = []
                for ds in ds_ranges:
                    vec = data.get(ds, [0] * behaviors_portrait_dim)
                    vecs.append(vec)
                role_behaviors_portraits.append(vecs)
            else:
                role_behaviors_portraits.append([[0] * behaviors_portrait_dim for _ in ds_ranges])
        role_behaviors_portraits = np.asarray(role_behaviors_portraits)

    gc.collect()

    X1 = np.concatenate((role_origin_portraits, role_behaviors_portraits), axis=-1)
    print(role_origin_portraits.shape)
    print(role_behaviors_portraits.shape)
    print(X1.shape)  # [num_role_ids, num_ds, feature_dim]
    num_role_ids, num_ds, feature_dim = X1.shape

    X1 = X1.reshape([-1, feature_dim])
    scaler = StandardScaler(copy=False)
    scaled_X1 = scaler.fit_transform(X1)
    scaled_X1 = scaled_X1.reshape([num_role_ids, num_ds, feature_dim])

    scaler_path = os.path.join(train_dir, str(server), 'StandardScaler_X1_{}.scaler'.format(model_tag))
    joblib.dump(scaler, scaler_path)

    with open(os.path.join(train_dir, str(server), 'scaled_X1_{}.pkl'.format(model_tag)), 'wb') as fout:
        pickle.dump(scaled_X1, fout)
    with open(os.path.join(train_dir, str(server), 'y.pkl'), 'wb') as fout:
        pickle.dump(y, fout)
    gc.collect()


def get_test_dataset(server, behaviors_portrait_dict, origin_portrait_dict, model_tag):
    with open(os.path.join(test_dir, str(server), 'pos_role_ids_test.json'), 'r', encoding='utf-8') as fin:
        pos_role_ids = json.load(fin)
    with open(os.path.join(test_dir, str(server), 'neg_role_ids_test.json'), 'r', encoding='utf-8') as fin:
        neg_role_ids = json.load(fin)

    role_ids = pos_role_ids + neg_role_ids
    y = [1] * len(pos_role_ids) + [0] * len(neg_role_ids)

    role_origin_portraits = []
    for role_id in tqdm.tqdm(role_ids, desc="role_origin_portraits"):
        if role_id in origin_portrait_dict:
            data = origin_portrait_dict[role_id]
            vecs = []
            for ds in ds_ranges:
                vec = data.get(ds, [0] * origin_portrait_dim)
                vecs.append(vec)
            role_origin_portraits.append(vecs)
        else:
            role_origin_portraits.append([[0] * origin_portrait_dim for _ in ds_ranges])
    role_origin_portraits = np.asarray(role_origin_portraits)

    if behaviors_portrait_dict == {}:
        role_behaviors_portraits = np.zeros(shape=(role_origin_portraits.shape[0],
                                                   role_origin_portraits.shape[1],
                                                   256), dtype=np.float32)
    else:
        role_behaviors_portraits = []
        for role_id in tqdm.tqdm(role_ids, desc="role_behaviors_portraits"):
            if role_id in behaviors_portrait_dict:
                data = behaviors_portrait_dict[role_id]
                vecs = []
                for ds in ds_ranges:
                    vec = data.get(ds, [0] * behaviors_portrait_dim)
                    vecs.append(vec)
                role_behaviors_portraits.append(vecs)
            else:
                role_behaviors_portraits.append([[0] * behaviors_portrait_dim for _ in ds_ranges])
        role_behaviors_portraits = np.asarray(role_behaviors_portraits)

    gc.collect()

    X1 = np.concatenate((role_origin_portraits, role_behaviors_portraits), axis=-1)

    scaler_path = os.path.join(train_dir, str(server), 'StandardScaler_X1_{}.scaler'.format(model_tag))
    scaler = joblib.load(scaler_path)
    num_role_ids, num_ds, feature_dim = X1.shape
    X1 = X1.reshape([-1, feature_dim])
    scaled_X1 = scaler.transform(X1)
    scaled_X1 = scaled_X1.reshape([num_role_ids, num_ds, feature_dim])

    with open(os.path.join(test_dir, str(server), 'scaled_X1_{}.pkl'.format(model_tag)), 'wb') as fout:
        pickle.dump(scaled_X1, fout)
    with open(os.path.join(test_dir, str(server), 'y.pkl'), 'wb') as fout:
        pickle.dump(y, fout)
    gc.collect()


if __name__ == "__main__":
    import sys

    assert len(sys.argv) in (2, 3)

    server = sys.argv[1]
    model_tag = sys.argv[2]

    origin_portrait_dict = {}
    origin_portrait_dim = None
    with open(os.path.join(base_dir, '{}.pos.portraits'.format(server)), 'r') as pos_fin, \
            open(os.path.join(base_dir, '{}.neg.portraits'.format(server)), 'r') as neg_fin:
        for l in tqdm.tqdm(pos_fin, desc='{}.pos.portraits'.format(server)):
            try:
                role_id, ds, *portrait_features = l.strip().split(',')
                role_id = int(role_id)
                if role_id not in origin_portrait_dict:
                    origin_portrait_dict[role_id] = {}
                origin_portrait_dict[role_id][ds] = list(map(float, portrait_features))
                if origin_portrait_dim is None:
                    origin_portrait_dim = len(origin_portrait_dict[role_id][ds])
                else:
                    assert origin_portrait_dim == len(origin_portrait_dict[role_id][ds]), \
                        "origin_portrait_dim={}, len(origin_portrait_dict[role_id][ds])={}, role_id={}, ds={}".format(
                            origin_portrait_dim, len(origin_portrait_dict[role_id][ds]), role_id, ds
                        )
            except:
                print(l)
                raise

        for l in tqdm.tqdm(neg_fin, desc='{}.neg.portraits'.format(server)):
            role_id, ds, *portrait_features = l.strip().split(',')
            role_id = int(role_id)
            if role_id not in origin_portrait_dict:
                origin_portrait_dict[role_id] = {}
            origin_portrait_dict[role_id][ds] = list(map(float, portrait_features))
            assert len(origin_portrait_dict[role_id][ds]) == origin_portrait_dim

    behaviors_portrait_dict = {}
    if model_tag != 'empty':
        behaviors_portrait_dim = None
        with open(os.path.join(base_dir, '{}.pos.behaviors_vectors_{}'.format(server, model_tag)), 'r') as pos_fin, \
                open(os.path.join(base_dir, '{}.neg.behaviors_vectors_{}'.format(server, model_tag)), 'r') as neg_fin:
            for l in tqdm.tqdm(pos_fin, desc='{}.pos.behaviors_vectors_{}'.format(server, model_tag)):
                role_id, ds, *portrait_features = l.strip().split(',')
                role_id = int(role_id)
                if role_id not in behaviors_portrait_dict:
                    behaviors_portrait_dict[role_id] = {}
                behaviors_portrait_dict[role_id][ds] = list(map(float, portrait_features))
                if behaviors_portrait_dim is None:
                    behaviors_portrait_dim = len(behaviors_portrait_dict[role_id][ds])
                else:
                    assert behaviors_portrait_dim == len(behaviors_portrait_dict[role_id][ds]), \
                        "behaviors_portrait_dim={}, behaviors_portrait_dict[role_id][ds]={} role_id={}, ds={}".format(
                            behaviors_portrait_dim, len(behaviors_portrait_dict[role_id][ds]), role_id, ds
                        )

            for l in tqdm.tqdm(neg_fin, desc='{}.neg.behaviors_vectors_{}'.format(server, model_tag)):
                role_id, ds, *portrait_features = l.strip().split(',')
                role_id = int(role_id)
                if role_id not in behaviors_portrait_dict:
                    behaviors_portrait_dict[role_id] = {}
                behaviors_portrait_dict[role_id][ds] = list(map(float, portrait_features))
                assert behaviors_portrait_dim == len(behaviors_portrait_dict[role_id][ds])

    split_to_train_test(server)
    get_train_dataset(server=server,
                      behaviors_portrait_dict=behaviors_portrait_dict,
                      origin_portrait_dict=origin_portrait_dict,
                      model_tag=model_tag)
    get_test_dataset(server=server,
                     behaviors_portrait_dict=behaviors_portrait_dict,
                     origin_portrait_dict=origin_portrait_dict,
                     model_tag=model_tag)
