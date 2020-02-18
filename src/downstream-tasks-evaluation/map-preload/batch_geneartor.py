import pickle
import numpy as np
from datetime import datetime, timedelta
import keras.preprocessing.sequence as sequence
import pytz


def read_pkl(src_file):
    with open(src_file, 'rb') as f:
        return pickle.load(f)


def get_phase(ts, n_phase=12):
    cur_time = datetime.fromtimestamp(float(ts))
    cur_hour = int(cur_time.hour)
    return int(cur_hour // (24/n_phase))


def get_dates(ts, period_type='daily', n_period=6, shift_hour=6):
    assert period_type in ['daily', 'weekly']
    tzname = pytz.timezone('Asia/Shanghai')
    cur_time = datetime.fromtimestamp(float(ts), tz=tzname)
    cur_time = cur_time - timedelta(hours=shift_hour)  # MMORPG player cycle starts at 6AM everyday
    dates = []
    _period = 1
    if period_type == 'weekly':
        _period = 7
    for i in range(n_period):
        date = (cur_time - timedelta(days=(i+1)*_period)).strftime('%F')
        dates = [date] + dates
    return dates


class BatchGenerator:
    def __init__(self, hp, make_portraits_zero=False):
        # data holder
        self.x_train = read_pkl(hp.X_file_train)  # map_id input
        self.x_test = read_pkl(hp.X_file_test)  # map_id input
        self.y_train = read_pkl(hp.y_file_train)    # map_id output
        self.y_test = read_pkl(hp.y_file_test)  # map_id output
        self.time_train = read_pkl(hp.Time_file_train)  # timestamp input
        self.time_test = read_pkl(hp.Time_file_test)    # timestamp input
        self.time_label_train = read_pkl(hp.Time_file_train_label)  # timestamp output
        self.time_label_test = read_pkl(hp.Time_file_test_label)  # timestamp output

        self.portrait_train = read_pkl(hp.portrait_train)   # portrait input
        self.portrait_test = read_pkl(hp.portrait_test)  # portrait input

        if make_portraits_zero:
            self.portrait_train = np.zeros_like(self.portrait_train)
            # print("============", self.portrait_train.shape)
            self.portrait_test = np.zeros_like(self.portrait_test)
            # print("============", self.portrait_test.shape)

        self.role_id_train = read_pkl(hp.role_id_train)  # role_id input
        self.role_id_test = read_pkl(hp.role_id_test)   # role_id input

        # parameter
        self.batch_size = hp.batch_size
        self.hp = hp

    def get_batch(self, datatype):
        if datatype == 'training':  # generate training batch
            x_ndarray = np.array(self.x_train)
            y_ndarray = np.asarray(self.y_train, dtype=int)
            time_ndarray = np.array(self.time_train)
            time_label_ndarray = np.array(self.time_label_train)
            portrait_ndarray = np.array(self.portrait_train)
            role_id = np.asarray(self.role_id_train, dtype=int)
        else:   # generate test batch
            x_ndarray = np.array(self.x_test)
            y_ndarray = np.asarray(self.y_test, dtype=int)
            time_ndarray = np.array(self.time_test)
            time_label_ndarray = np.array(self.time_label_test)
            portrait_ndarray = np.array(self.portrait_test)
            role_id = np.asarray(self.role_id_test, dtype=int)

        pos = 0
        n_records = len(y_ndarray)
        shuffle_index0 = np.random.permutation(n_records)  # global shuffling

        while True:  # generator shuffled batch data
            st = pos
            ed = pos + self.batch_size
            x_batch = x_ndarray[shuffle_index0][st:ed]
            y_batch = y_ndarray[shuffle_index0][st:ed]
            time_batch = time_ndarray[shuffle_index0][st:ed]
            time_label_batch = time_label_ndarray[shuffle_index0][st:ed]
            portrait_batch = portrait_ndarray[shuffle_index0][st:ed]
            role_id_batch = role_id[shuffle_index0][st:ed]

            # padding
            x_batch = sequence.pad_sequences(x_batch, maxlen=self.hp.maxlen, dtype='int32',
                                             padding='pre', truncating='pre', value=0)
            time_batch = sequence.pad_sequences(time_batch, maxlen=self.hp.maxlen, dtype='int32',
                                                padding='pre', truncating='pre', value=0)
            pos += self.batch_size
            if pos >= n_records:
                pos = 0
                shuffle_index0 = np.random.permutation(n_records)  # global reshuffling
            shuffle_index1 = np.random.permutation(len(y_batch))  # local shuffling
            yield x_batch[shuffle_index1], y_batch[shuffle_index1], time_batch[shuffle_index1], \
                time_label_batch[shuffle_index1], portrait_batch[shuffle_index1], role_id_batch[shuffle_index1]
