#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, normalize

train_dir = '/project/fumo/gameBERT/nsh_measurement/map_preload/train'
test_dir = '/project/fumo/gameBERT/nsh_measurement/map_preload/test'
base_dir = '/project/fumo/gameBERT/nsh_measurement/map_preload'


def read_pkl(src_file):
    with open(src_file, 'rb') as f:
        return pickle.load(f)


def __portrait_phase_dict(phases=12):
    p_dict = {}
    p_interval = int(24/phases)
    for weekday in range(7): # from 0 to 6
        p_dict[weekday] = {}
        for i in range(phases):
            for j in range(i*p_interval, i*p_interval+p_interval):
                p_dict[weekday][j] = i + weekday*phases
    return p_dict


def __test_portrait_process(portrait_df, norms):
    df = pd.DataFrame(portrait_df['portrait_features'].tolist(), index=portrait_df.index)
    df = df.fillna(0)
    # print(norms)
    value_normalized = df.values.astype(float) / norms
    df_n = pd.DataFrame(value_normalized, columns=df.columns, index=df.index)
    return df_n


def __train_portrait_process(portrait_df):
    df = pd.DataFrame(portrait_df['portrait_features'].tolist(), index=portrait_df.index)
    df = df.fillna(0)
    value_normalized, feature_nroms = normalize(df.values.astype(float), axis=0, return_norm=True)
    df_n = pd.DataFrame(value_normalized, columns=df.columns, index=df.index)
    # print('Finished normalization.\n')
    return df_n, feature_nroms


def __test_portrait_combination(role_id_seq, seq_time_label, portrait_df, portrait_len=256, time_phase_len=24*7):
    portrait_vectors = np.zeros((len(seq_time_label), portrait_len))

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
    onehot_encoder.fit(np.arange(time_phase_len).reshape(-1, 1))

    time_phase_dict = __portrait_phase_dict(int(time_phase_len / 7))

    time_phases = []
    for time_label in seq_time_label:
        t_time = datetime.fromtimestamp(float(time_label))
        t_phase = time_phase_dict.get(t_time.weekday(), 0).get(t_time.hour, 0)
        time_phases.append(t_phase)
    time_phases = np.asarray(time_phases).reshape(-1, 1)

    phase_vectors = onehot_encoder.transform(time_phases)

    for i in range(len(seq_time_label)):
        date = datetime.fromtimestamp(float(seq_time_label[i]))
        date_str = date.strftime('%F')
        role_id = role_id_seq[i]
        try:
            portrait_vec = portrait_df.loc[(date_str, role_id)].values.astype(float)
        except Exception as e:
            # print(role_id, date)
            portrait_vec = np.zeros(portrait_len, np.float32)
        portrait_vectors[i, :] = portrait_vec

    portrait_list = np.concatenate((portrait_vectors, phase_vectors), axis=1)
    return portrait_list


def __train_portrait_combination(role_id_seq, seq_time_label, portrait_df, portrait_len=256, time_phase_len=24*7):
    portrait_vectors = np.zeros((len(seq_time_label), portrait_len))

    onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
    onehot_encoder.fit(np.arange(time_phase_len).reshape(-1, 1))

    portrait_time_phase_dict = __portrait_phase_dict(int(time_phase_len / 7))

    time_phases = []
    for time_label in seq_time_label:
        t_time = datetime.fromtimestamp(float(time_label))
        t_phase = portrait_time_phase_dict.get(t_time.weekday(), 0).get(t_time.hour, 0)
        time_phases.append(t_phase)
    time_phases = np.asarray(time_phases).reshape(-1, 1)

    phase_vectors = onehot_encoder.transform(time_phases)

    for i in range(len(seq_time_label)):
        date = datetime.fromtimestamp(float(seq_time_label[i]))
        date_str = date.strftime('%F')
        role_id = role_id_seq[i]
        try:
            portrait_vec = portrait_df.loc[(date_str, role_id)].values.astype(float)
        except Exception as e:
            portrait_vec = np.zeros(portrait_len, np.float32)
        portrait_vectors[i, :] = portrait_vec

    portrait_list = np.concatenate((portrait_vectors, phase_vectors), axis=1)
    return portrait_list


def get_train_portrait_df_from_local_behaviors_text(server, model_tag, behaviors_vector_dim=256):

    # df.set_index(['ds', 'role_id'], inplace=True)
    # columns = ['ds', 'role_id', 'portrait_features']
    # contents = ['2019-12-14', '12345678', '0.1|0.2|...|0.3']
    with open(os.path.join(base_dir, "{}.{}.behaviors_vectors".format(server, model_tag)), 'r') as fin:
        res = []
        for l in fin:
            role_id, ds, *portrait_features = l.strip().split(',')
            res.append({'ds': ds, 'role_id': role_id, 'portrait_features': portrait_features})
        portrait_df = pd.DataFrame(res)

    portrait_df, portrait_feature_norms = __train_portrait_process(portrait_df)

    role_id_file_train = os.path.join(train_dir, str(server), 'all_role_id_train_seq11.pkl')
    role_id_seq = read_pkl(role_id_file_train)
    role_id_seq = role_id_seq.tolist()

    Time_file_train_label = os.path.join(train_dir, str(server), 'all_timetrain_label_seq11.pkl')
    y_time_label_train = read_pkl(Time_file_train_label)
    y_time_list = y_time_label_train.tolist()

    portrait_list = __train_portrait_combination(role_id_seq, y_time_list, portrait_df, behaviors_vector_dim)

    portrait_file_train = os.path.join(train_dir, str(server), 'all_portrait_train_seq11_{}.pkl'.format(model_tag))
    with open(portrait_file_train, 'wb') as f:
        pickle.dump(portrait_list, f)
    portrait_feature_norms_file = os.path.join(train_dir, str(server), 'portrait_features_norms_{}.pkl'.format(model_tag))
    with open(portrait_feature_norms_file, 'wb') as f:
        pickle.dump(portrait_feature_norms, f)


def get_test_portrait_df_from_local_behaviors_text(server, model_tag, behaviors_vector_dim=256):
    with open(os.path.join(base_dir, "{}.{}.behaviors_vectors".format(server, model_tag)), 'r') as fin:
        res = []
        for l in fin:
            role_id, ds, *portrait_features = l.strip().split(',')
            res.append({'ds': ds, 'role_id': role_id, 'portrait_features': portrait_features})
        portrait_df = pd.DataFrame(res)

    portrait_feature_norms_file = os.path.join(train_dir, str(server), 'portrait_features_norms_{}.pkl'.format(model_tag))
    portrait_feature_norms = read_pkl(portrait_feature_norms_file)

    portrait_df = __test_portrait_process(portrait_df, portrait_feature_norms)

    role_id_file_train = os.path.join(test_dir, str(server), 'all_role_id_train_seq11.pkl')
    role_id_seq = read_pkl(role_id_file_train)
    role_id_seq = role_id_seq.tolist()
    Time_file_train_label = os.path.join(test_dir, str(server), 'all_timetrain_label_seq11.pkl')
    y_time_list = read_pkl(Time_file_train_label)
    y_time_list = y_time_list.tolist()

    portrait_list = __test_portrait_combination(role_id_seq, y_time_list, portrait_df, behaviors_vector_dim)
    portrait_file_train = os.path.join(test_dir, str(server), 'all_portrait_train_seq11_{}.pkl'.format(model_tag))
    with open(portrait_file_train, 'wb') as f:
        pickle.dump(portrait_list, f)


def make_copy_of_train_test_portraits(server, from_model_tag):
    import shutil

    test_portrait = os.path.join(test_dir, str(server), 'all_portrait_train_seq11_{}.pkl')
    train_portrait = os.path.join(train_dir, str(server), 'all_portrait_train_seq11_{}.pkl')

    src_test_portrait = test_portrait.format(from_model_tag)
    src_train_portrait = train_portrait.format(from_model_tag)

    shutil.copy(src_test_portrait, test_portrait.format('empty'))
    shutil.copy(src_train_portrait, train_portrait.format('empty'))


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3

    server, model_tag = sys.argv[1], sys.argv[2]

    if model_tag == 'empty':
        make_copy_of_train_test_portraits(server=int(server), from_model_tag='bert64')
    else:
        get_train_portrait_df_from_local_behaviors_text(server=int(server),
                                                        model_tag=model_tag,
                                                        behaviors_vector_dim=256)
        get_test_portrait_df_from_local_behaviors_text(server=int(server),
                                                       model_tag=model_tag,
                                                       behaviors_vector_dim=256)
    print("data preprocess done. server={}, model_tag={}".format(server, model_tag))
