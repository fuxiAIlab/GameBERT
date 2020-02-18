#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras.backend as K
import keras.layers
from keras.layers import Permute, Multiply, Dropout, Dense
from keras.models import Input, Model
from keras.utils import multi_gpu_model
from tcn import TCN


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(30, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def tilted_loss(q, y, f):
    e = (y - f)
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)


def f1_score(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = c1 / (c2 + K.epsilon())
    recall = c1 / (c3 + K.epsilon())
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score


def prec(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = c1 / (c2 + K.epsilon())
    return precision


def recal(y_true, y_pred):
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = c1 / (c3 + K.epsilon())
    return recall


def train_model(X1_train, X1_test, y_train, y_test, lr, epochs, num_gpus, batch_size, feature_dims, use_multi_gpu):

    inputs1 = Input(shape=(30, feature_dims))
    x1 = Permute((2, 1))(inputs1)
    x1 = keras.layers.Conv1D(32, 1, padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=keras.regularizers.l2(0.01))(x1)
    x1 = Permute((2, 1))(x1)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)

    x2 = TCN(nb_filters=32, kernel_size=2, return_sequences=True, use_batch_norm=True, dropout_rate=0.2)(inputs1)
    x2 = attention_3d_block(x2)
    x2 = keras.layers.GlobalMaxPooling1D()(x2)
    x = keras.layers.Concatenate()([x1, x2])

    x = keras.layers.normalization.BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.normalization.BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    output = Dense(1, activation='relu', name='output')(x)

    model = Model(inputs=[inputs1], outputs=[output])
    if use_multi_gpu:
        model = multi_gpu_model(model, gpus=num_gpus)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr, clipnorm=1.),
                  loss=lambda y, f: tilted_loss(0.5, y, f),
                  metrics=['mae', f1_score, prec, recal])
    model.fit([X1_train], y_train, epochs=epochs,
              batch_size=batch_size * num_gpus,
              validation_data=([X1_test], y_test),
              verbose=2)
