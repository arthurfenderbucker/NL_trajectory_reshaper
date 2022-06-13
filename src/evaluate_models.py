import numpy as np
# import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import os
import json
from functions import *
import datetime

import argparse

import absl.logging #prevent checkpoint warnings while training
# absl.logging.set_verbosity(absl.logging.ERROR)
import time
from motion_refiner_4D import Motion_refiner
# from mlflow.tracking import MlflowClient
# import mlflow


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--dataset_dir', default='/home/tum/data/data/',
                    help='Dataset directory.')
parser.add_argument('--models_path', default="/home/tum/data/models/")
parser.add_argument('--exp_name', default="experimet_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

parser.add_argument('--lr', type=float, default=0.0) #use default lr decay
parser.add_argument('--num_enc', type=int, default=1)
parser.add_argument('--num_dec', type=int, default=5)
parser.add_argument('--num_dense', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--model_depth', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dff', type=int, default=512)
parser.add_argument('--dense_n', type=int, default=512)
parser.add_argument('--concat_emb', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--augment', type=int, default=0)
parser.add_argument('--base_model', type=str, default="None")
parser.add_argument('--refine', type=int, default=0)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--activation', type=str, default="linear")
parser.add_argument('--ignore_features', type=int, default=0)
parser.add_argument('--norm_layer', type=int, default=1)
parser.add_argument('--num_emb_vec', type=int, default=2)
parser.add_argument('--ds_size_factor', type=float, default=1.0)
parser.add_argument('--CLIP_only', type=int, default=0)

parser.add_argument('--augment_text', default="None")
parser.add_argument('--new_dataset_name', default="4D_1M_scaping_factor")
parser.add_argument('--dataset_name', default="4D_2to6_norm")


args = parser.parse_args()


concat_emb = False if args.concat_emb == 0 else True 
augment = False if args.augment == 0 else True 
ignore_features = False if args.ignore_features == 0 else True
norm_layer = False if args.norm_layer == 0 else True

delimiter ="-"




traj_n = 40
mr = Motion_refiner(load_models=False ,traj_n = traj_n,locality_factor=True)
feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()
embedding_indices = mr.embedding_indices
# embedding_indices = np.concatenate([feature_indices,obj_sim_indices,obj_poses_indices])


# dataset_name = "4D_10000_objs_2to6_norm_"
# dataset_name = "4D_100k_scaping_factor"
dataset_name = "4D_1M_scaping_factor"

X,Y, data = mr.load_dataset(dataset_name, filter_data = True, base_path=args.dataset_dir)
X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val = mr.split_dataset(X, Y, test_size=0.2, val_size=0.1)


X_train = X_train[:int(args.ds_size_factor*X_train.shape[0]),:]
y_train = y_train[:int(args.ds_size_factor*y_train.shape[0]),:]
indices_train = indices_train[:int(args.ds_size_factor*indices_train.shape[0])]


X_valid = X_valid[:int(args.ds_size_factor*X_valid.shape[0]),:]
y_valid = y_valid[:int(args.ds_size_factor*y_valid.shape[0]),:]
indices_valid = indices_val[:int(args.ds_size_factor*indices_val.shape[0])]


test_resize_factor = 0.1 #for faster evaluation
X_test = X_test[:int(test_resize_factor*X_test.shape[0]),:]
y_test = y_test[:int(test_resize_factor*y_test.shape[0]),:]
indices_test = indices_test[:int(test_resize_factor*indices_test.shape[0])]


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
# from keras import backend as K
# from keras.models import Sequential
# from keras import Model
# from keras.layers import BatchNormalization,GlobalAveragePooling1D, Embedding,Flatten, Layer,Dense,Dropout,MultiHeadAttention, Attention, Conv1D ,Input,Lambda, Concatenate,LayerNormalization
from tensorflow.python.client import device_lib
# from sklearn.model_selection import train_test_split

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print("\n\n devices: ",get_available_devices()) 

# tf.random.set_seed(42)

seed = 42
n_samples, input_size = X.shape
#------------------------------------------------------------------------


from TF4D_mult_features import *


features_n = len(embedding_indices)

#------------------------------------------------------------------------
#------------------------------------------------------------------------

bs=args.bs
def prepare_x(x):
  objs = pad_array(list_to_wp_seq(x[:,obj_poses_indices],d=3),4,axis=-1) # no speed
  trajs = list_to_wp_seq(x[:,traj_indices],d=4)
  return np.concatenate([objs,trajs],axis = 1)

train_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_train),
                                                  list_to_wp_seq(y_train,d=4),
                                                  X_train[:,embedding_indices])).batch(bs)
print("train dataset created")
val_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_valid),
                                                  list_to_wp_seq(y_valid,d=4),
                                                  X_valid[:,embedding_indices])).batch(bs)
print("validation dataset created")
test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
                                                  list_to_wp_seq(y_test,d=4),
                                                  X_test[:,embedding_indices])).batch(bs)
print("test dataset created")

num_batches = 1000
val_batches = 100


print("\nnum batches: train ",num_batches,"\tval ",val_batches)

models_path = os.path.join(args.models_path,args.exp_name+"/")
print("\nmodels_path: \t",models_path)


H = []
best_model = ""
min_val_loss = 100.0
models = {}

def evaluate_model(model, epoch):

    print("epoch:",epoch)
    # print("\nwith data augmentation:")
    # result_eval_aug = model.evaluate(generator(test_dataset,stop=True))[0]

    x_test_new, y_test_new = prepare_x(X_test), list_to_wp_seq(y_test,d=4)
    emb_test_new = X_test[:,embedding_indices]

    result_eval = model.evaluate((x_test_new, y_test_new[:,:-1,:], emb_test_new), y_test_new[:,1:,:])[0]

    print("\n ----------------------------------------")
    print("withdata generation:")
    test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
                                                    list_to_wp_seq(y_test,d=4),
                                                    X_test[:,embedding_indices])).batch(X_test.shape[0])

    g = generator(test_dataset,stop=True,augment=False)
    x_t, y_t = next(g)
    pred = generate(model ,x_t, traj_n=traj_n).numpy()
    print(pred.shape)
    result_gen = np.average((y_t - pred[:,1:,:])**2)
    print("Test loss w generation: ",result_gen)



    # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    # with file_writer.as_default():
    #     tf.summary.scalar('test_result_gen', data=result_gen, step=epoch)
    #     tf.summary.scalar('test_result_eval', data=result_eval, step=epoch)



# set_folder = args.set_name

models_path = [os.path.join(models_path,d) for d in os.listdir(models_path) if d[-3:]==".h5"]

for i,model_file in enumerate(models_path):
    

    print(model_file)
    model = load_model(model_file,delimiter=delimiter)

    total_epochs = model.optimizer.iterations.numpy() // num_batches
    evaluate_model(model,epoch = total_epochs+1)


