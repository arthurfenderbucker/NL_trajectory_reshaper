import numpy as np
import matplotlib.pyplot as plt
import os
import json
import random
import re
from functions import *
import random
import datetime

import argparse

import absl.logging #prevent checkpoint warnings while training
absl.logging.set_verbosity(absl.logging.ERROR)

from motion_refiner import Motion_refiner



parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--dataset_dir', default='/home/tum/data/data/',
                    help='Dataset directory.')
parser.add_argument('--models_path', default="/home/tum/data/models/TF_augmented_data/")
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_enc', type=int, default=2)
parser.add_argument('--num_dec', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--model_depth', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dff', type=int, default=512)

args = parser.parse_args()


traj_n = 10
mr = Motion_refiner(traj_n = traj_n)

## ------- processed data -------
# X,Y = mr.prepare_data(data,deltas=True)
# print("X: ",X.shape)
# print("Y: ",Y.shape)

## ------- save pre processed data -------
# mr.save_XY(X, Y, x_name="X_delta_new_names",y_name="Y_delta_new_names")
# mr.save_data(data,data_name="data_delta_new_names")

# ------- load data --------
print("loading data...")

X_, Y_ = mr.load_XY(x_name="X_delta_new_names",y_name="Y_delta_new_names", base_path=args.dataset_dir)
data_ = mr.load_data(data_name="data_delta_new_names", base_path=args.dataset_dir)
feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()


#------------------------------------------------------------------------

# print(reorganize_input(np.expand_dims(np.array(range(20)),axis=0)))

Y_abs = Y_+ X_[:,traj_indices]

def has_word(t, words):
    for w in words:
        if w in t:
            return False
    return True
def filter_cartesian(x, y, data, word=["left","right","front","back"]):
    # filters samples where the out traj is not between 0 and 1

    
    i_invalid = np.array([i for i,d in enumerate(data) if has_word(d["text"], words) ])
    y_filtered = np.delete(y, i_invalid, axis=0)
    X_filtered = np.delete(x, i_invalid, axis=0)
    i_valid = np.delete(np.arange(len(y)), i_invalid, axis=0)
    data_filtered = [data[i] for i in i_valid]
    return X_filtered, y_filtered, data_filtered, i_invalid


X,Y, data = X_,Y_abs,data_
# X,Y, data, i_invalid = filter(X_,Y_abs,data_) #for delta predictions
# X,Y, data, i_invalid = filter_cartesian(X_,Y_abs,data_) #for delta predictions

# X,Y, data, i_invalid = filter(X_,Y_abs,data_,lower_limit=0) #for wp predictions

print("X:",X_.shape,"\tY:",Y_.shape)
print("filtered X:",X.shape,"\tY:",Y.shape)

print(limits(X[:,traj_indices]))
print(limits(Y))

#------------------------------------------------------------------------


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from keras import backend as K
from keras.models import Sequential
from keras import Model
from keras.layers import BatchNormalization,GlobalAveragePooling1D, Embedding,Flatten, Layer,Dense,Dropout,MultiHeadAttention, Attention, Conv1D ,Input,Lambda, Concatenate,LayerNormalization
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 

tf.random.set_seed(42)

print("X:",X.shape,"\tY:",Y.shape)
# print("filtered: ", len(i_invalid))

# Split the data: 70% train 20% test 10% validation
n_samples, input_size = X.shape # 768+traj_n*2+max_num_objs*3
X_train_, X_test, y_train_, y_test, indices_train_, indices_test= train_test_split(X, Y,np.arange(n_samples), test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid, indices_train, indices_val = train_test_split(X_train_, y_train_, indices_train_ ,test_size=0.125, shuffle= False)
print("Train X:",X_train.shape,"\tY:",y_train.shape)
print("Test  X:",X_test.shape,"\tY:",y_test.shape)
print("Val   X:",X_valid.shape,"\tY:",y_valid.shape)


#------------------------------------------------------------------------


from simple_TF_continuos import *

embedding_indices = np.concatenate([feature_indices,obj_sim_indices,obj_poses_indices])
# embedding_indices = np.concatenate([feature_indices,obj_sim_indices])

features_n = len(embedding_indices)


#------------------------------------------------------------------------


#------------------------------------------------------------------------

bs=args.bs
def prepare_x(x):
  objs = list_to_wp_seq(x[:,obj_poses_indices])
  trajs = list_to_wp_seq(x[:,traj_indices])
  return np.concatenate([objs,trajs],axis = 1)


def increase_dataset(x,y,embedding_indices,augment_factor):
    x_, y_ = prepare_x(x), list_to_wp_seq(y)
    emb = x[:,embedding_indices]

    x_new = x_
    y_new = y_
    emb_new=emb
    for i in range(augment_factor):
        x_new_i, y_new_i = augment_xy(x_,y_,width_shift_range=0.5, height_shift_range=0.5,rotation_range=np.pi,
                        zoom_range=[0.5,1.5],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
        x_new = np.append(x_new,x_new_i, axis=0)
        y_new = np.append(y_new,y_new_i, axis=0)
        emb_new = np.append(emb_new,emb, axis=0)

    print("new data shape: x=",x_new.shape,"   y=",y_new.shape,"   emb=", emb_new.shape)
    return x_new, y_new, emb_new




train_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_train),
                                                  list_to_wp_seq(y_train),
                                                  X_train[:,embedding_indices])).batch(bs)
val_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_valid),
                                                  list_to_wp_seq(y_valid),
                                                  X_valid[:,embedding_indices])).batch(bs)
test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
                                                  list_to_wp_seq(y_test),
                                                  X_test[:,embedding_indices])).batch(bs)

num_batches = 0
for (batch, (_,_,_)) in enumerate(train_dataset):
  num_batches = batch

val_batches = 0
for (batch, (_,_,_)) in enumerate(val_dataset):
  val_batches = batch

print(num_batches,val_batches)

def generator(data_set,stop=False,augment=True):

    while True:
        for x, y,emb in data_set:
            x_new, y_new = x,y
            if augment:
                x_new, y_new = augment_xy(x,y,width_shift_range=0.5, height_shift_range=0.5,rotation_range=np.pi,
                        zoom_range=[0.5,1.5],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
            else:
                x_new, y_new = augment_xy(x,y,width_shift_range=0.0, height_shift_range=0.0,rotation_range=0.0,
                        zoom_range=0.0,horizontal_flip=False, vertical_flip=False, offset=[-0.5,-0.5])

            yield ( [x_new , y_new[:, :-1],emb] , y_new[:, 1:] )
        if stop:
            break

# g = generator(train_dataset)


embedding_indices = np.concatenate([feature_indices,obj_sim_indices])


x_new, y_new, emb_new= increase_dataset(X_train ,y_train,embedding_indices,10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_new, y_new, emb_new)).batch(bs)
g = generator(train_dataset, augment=False)
source, y = next(g)