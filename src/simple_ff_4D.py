from keras.layers import BatchNormalization, GlobalAveragePooling1D, Embedding, Flatten, Layer, Dense, Dropout, MultiHeadAttention, Attention, Conv1D, Input, Lambda, Concatenate, LayerNormalization
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras import Model
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import re

import os

# def get_model(features_n=777,input_size=797, num_layers_enc=2, num_layers_dec=2,num_dense=3,dense_n=256, d_model=128, dff=512, num_heads=8, dropout_rate=0.1, wp_d=2,bs=32, concat_emb=False, optimizer="adam",norm_layer=True,activation="tanh"):

def get_model(input_size=797, features_n=777, num_layers_enc=2, num_layers_dec=2,num_dense=3,dense_n=256, d_model=128, dff=512, num_heads=8,
             dropout_rate=0.1, wp_d=4,bs=32,ds_size_factor=1.0,augment=0,traj_n=40,
                concat_emb=False, optimizer="adam",norm_layer=True,activation="linear", max_traj_len = 100, num_emb_vec=4):

    model = Sequential()
    model.add(Dense(dense_n*2, input_dim=input_size, activation='relu'))
    model.add(Dropout(dropout_rate))
    for i in range(num_dense):
        model.add(Dense(dense_n, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(traj_n*wp_d, activation=activation))
    compile(model, optimizer=optimizer)

    return model


def compile(model, optimizer="adam"):
    # optimizer = tf.keras.optimizers.Adam(1e-2, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-9)
    # optimizer = tf.keras.optimizers.Adam(1e-4)
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none')
    # optimizer = tf.keras.optimizers.Adam()
    MSE = tf.keras.losses.MeanSquaredError()
    # MSLE = tf.keras.losses.MeanSquaredLogarithmicError()
    epslon = 10e-3
    alpha = 0.01

    def custom_loss(y_true, y_pred):

        # maxlen = tf.shape(y_true)[1]
        # loss_mse = MSE(y_true, y_pred)
        # maxlen = tf.cast(maxlen, dtype=loss_mse.dtype)
        loss_mse = tf.reduce_mean(tf.square(y_true-y_pred))
        loss_log = tf.reduce_mean(tf.square(y_true-y_pred))
        _loss = tf.add(loss_mse, tf.math.multiply(alpha, loss_log))
        return loss_log
    loss = MSE

    def masked_loss(y_true, y_pred):
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        _loss = loss(y_true, y_pred)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask

        return tf.reduce_sum(_loss)/tf.reduce_sum(mask)
    metrics = [loss]  # , masked_loss]
    model.compile(optimizer=optimizer, loss=MSE, metrics=metrics)


def file_name2dict(s, delimiter ="&"):
    d = {}
    if s[-3:] == ".h5":
        s = s[:-3]
    for i in s.split(delimiter)[1:]:
        l = i.split(":")
        if '.' in l[1]:
            d[l[0]] = float(l[1])
        elif l[1] == "False":
            d[l[0]] = False
        elif l[1] == "True":
            d[l[0]] = True
        else:
            try:
                d[l[0]] = int(l[1])
            except:
                d[l[0]] = l[1]
    print(d)
    return d


def load_model(model_file, model_path="models/", delimiter ="&"):
    # file nam eexample: TF&num_layers_enc:2&num_layers_dec:2&d_model:128&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2.h5
    
    file_name = model_file.split("/")[-1]
    param = file_name2dict(file_name,delimiter=delimiter)
    model = get_model(**param)
    # model.load_weights(os.path.join(model_path,f))

    print("loading weights: ",model_file)
    model.load_weights(model_file)

    return model
