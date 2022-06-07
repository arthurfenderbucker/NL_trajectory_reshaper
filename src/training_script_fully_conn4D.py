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
absl.logging.set_verbosity(absl.logging.ERROR)

from motion_refiner_4D import Motion_refiner

reset_seed()

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--dataset_dir', default='/home/tum/data/data/',
                    help='Dataset directory.')
parser.add_argument('--models_path', default="/home/tum/data/models/")
parser.add_argument('--exp_name', default="experimet_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

parser.add_argument('--lr', type=float, default=0.0) #use default lr decay
parser.add_argument('--num_enc', type=int, default=1)
parser.add_argument('--num_dec', type=int, default=6)
parser.add_argument('--num_dense', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--model_depth', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dff', type=int, default=512)
parser.add_argument('--dense_n', type=int, default=512)
parser.add_argument('--concat_emb', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--augment', type=int, default=1)
parser.add_argument('--base_model', type=str, default="None")
parser.add_argument('--refine', type=int, default=1)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--activation', type=str, default="tanh")
parser.add_argument('--ignore_features', type=int, default=0)
parser.add_argument('--norm_layer', type=int, default=1)



args = parser.parse_args()



concat_emb = False if args.concat_emb == 0 else True 
augment = False if args.augment == 0 else True 
ignore_features = False if args.ignore_features == 0 else True
norm_layer = False if args.norm_layer == 0 else True

delimiter ="&"



traj_n = 40
mr = Motion_refiner(load_models=True ,traj_n = traj_n)
feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()
embedding_indices = np.concatenate([feature_indices,obj_sim_indices, obj_poses_indices])

features_n = len(embedding_indices)


dataset_name = "4D_10000_objs_2to6_norm_"
X,Y, data = mr.load_dataset(dataset_name, filter_data = True)
X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val = mr.split_dataset(X, Y, test_size=0.2, val_size=0.1)

n_samples, input_size = X_train.shape

if ignore_features:
    X[:,feature_indices] = 0
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!         IGNORING FEATURES          !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


from simple_ff_4D import *


bs=args.bs
def prepare_x(x):
  objs = pad_array(list_to_wp_seq(x[:,obj_poses_indices],d=3),4,axis=-1)
  trajs = list_to_wp_seq(x[:,traj_indices],d=4)
  return np.concatenate([objs,trajs],axis = 1)

train_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_train),
                                                  list_to_wp_seq(y_train,d=4),
                                                  X_train[:,embedding_indices])).batch(bs)

val_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_valid),
                                                  list_to_wp_seq(y_valid,d=4),
                                                  X_valid[:,embedding_indices])).batch(bs)

test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
                                                  list_to_wp_seq(y_test,d=4),
                                                  X_test[:,embedding_indices])).batch(bs)

num_batches = 0
for (batch, (_,_,_)) in enumerate(train_dataset):
    num_batches = batch

val_batches = 0
for (batch, (_,_,_)) in enumerate(val_dataset):
    val_batches = batch

print("\nnum batches: train ",num_batches,"\tval ",val_batches)

models_path = os.path.join(args.models_path,args.exp_name+"/")
print("\nmodels_path: \t",models_path)


H = []
best_model = ""
min_val_loss = 100.0
models = {}


param = dict(num_layers_enc = args.num_enc,
                num_layers_dec = args.num_dec,
                d_model = args.model_depth,
                dff = args.dff,
                input_size = 976,
                num_heads = args.num_heads,
                dropout_rate = args.dropout,
                wp_d=2,
                bs=args.bs,
                dense_n=args.dense_n,
                num_dense=args.num_dense,
                concat_emb=concat_emb,
                features_n=features_n,
                optimizer=args.optimizer,
                norm_layer=norm_layer,
                activation=args.activation)

param_s = json.dumps(param)
model_name = "FF4D"
for k,v in param.items():
    model_name+=delimiter+k+":"+str(v)
model_file = os.path.join(models_path,model_name+".h5")


if os.path.exists(model_file): #training on the same experiment with and existing models param
    args.base_model = model_file
    print("training on the same experiment with and existing models param")

if args.base_model != "None":
    print("\nloading base model...")
    print("base_model", args.base_model)
    
    if args.base_model[-3:] == ".h5": #base_model is a path

        model = load_model(args.base_model)
        base_model_name = args.base_model.split("/")[-1][:-3]
    else:
        base_model_name = args.base_model
        base_model_file = os.path.join(models_path,base_model_name+".h5")
        model = load_model(base_model_file)


    model_name = "refined_"+base_model_name #update the model name to avoid overwrite
    model_file = os.path.join(models_path,model_name+".h5") #new refined model file
    # compile(model)
    print("DONE")
else:
    # reset_logs(models_path+"logs")
    print("\ncreating new model...")

    model = get_model(**param)
    # compile(model)

print("model_file: \t",model_file)



logdir = os.path.join(models_path,"logs",model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(models_path):
    os.makedirs(models_path)


def linearize_data(source, y):
    source, y
    x,y_shifted,emb = source
    bs = x.shape[0]
    x_flat = np.concatenate([np.reshape(x,[bs,-1]),emb],axis=1)
    y_flat = np.concatenate([np.reshape(y_shifted[:,:1,:],[bs,-1]), np.reshape(y,[bs,-1])],axis=1)
    return (x_flat, y_flat)


def generator(data_set,stop=False,augment=True, num_objs = 3):

    while True:
        for x, y,emb in data_set:
            x_new, y_new = x,y
            # if augment:
            #     x_new, y_new = augment_xy(x,y,width_shift_range=0.3, height_shift_range=0.3,rotation_range=np.pi,
            #             zoom_range=[0.6,1.1],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
            # else:
            #     x_new, y_new = augment_xy(x,y,width_shift_range=0.0, height_shift_range=0.0,rotation_range=0.0,
            #             zoom_range=0.0,horizontal_flip=False, vertical_flip=False, offset=[-0.5,-0.5])
            # emb[:,-num_batches:] = tf.one_hot(tf.argmax(emb[:,-num_batches:],1),num_objs).numpy()
            # emb_new = tf.concat([emb[:,:-num_batches],tf.one_hot(tf.argmax(emb[:,-num_batches:],1),num_objs)],-1)
            
            yield linearize_data( [x_new , y_new[:, :-1],emb] , y_new[:, 1:] )
        if stop:
            break




def increase_dataset(x,y,embedding_indices,augment_factor):
    x_, y_ = prepare_x(x), list_to_wp_seq(y,d=4)
    emb = x[:,embedding_indices]

    x_new = x_
    y_new = y_
    emb_new=emb
    for i in range(augment_factor):
        x_new_i, y_new_i = augment_xy(x,y,width_shift_range=0.3, height_shift_range=0.3,rotation_range=np.pi,
                        zoom_range=[0.6,1.1],horizontal_flip=True, vertical_flip=True, offset=[-0.5,-0.5])
        x_new = np.append(x_new,x_new_i, axis=0)
        y_new = np.append(y_new,y_new_i, axis=0)
        emb_new = np.append(emb_new,emb, axis=0)

    print("new data shape: x=",x_new.shape,"   y=",y_new.shape,"   emb=", emb_new.shape)
    return x_new, y_new, emb_new



def evaluate_model(model, epoch):


    print("without data augmentation:")
    result_eval = model.evaluate(X_test ,y_test)[0]
    print(result_eval)

    print("\n ----------------------------------------")
    print("withdata generation:")



    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    with file_writer.as_default():
        tf.summary.scalar('test_result_eval', data=result_eval, step=epoch)



if args.base_model != "None":

    print("\nEvaluating base model...")
    total_epochs = model.optimizer.iterations.numpy() // num_batches
    evaluate_model(model, epoch = total_epochs)

print("\n\n ----------------------------------------")
print("-----           TRAINING             ----")
print(" ----------------------------------------")
print("starting: ",model_name )



earlly_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  mode='min', verbose=2, patience=30)
tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=0, save_format="tf",
                                                    monitor='val_loss', mode='min', save_best_only=True)
absl.logging.set_verbosity(absl.logging.ERROR)  #prevent checkpoint warnings while training
total_epochs = 0

def warmup(v,ep):
    return [(i,1) for i in np.linspace(v/ep, v, num=ep)]

warmup_epochs = 15
lr_schedule = [(1e-4,1),(5e-5,1),(1e-5,1)]
# lr_schedule = warmup(1e-4,warmup_epochs)+[(1e-4,1),(5e-5,1),(1e-5,1)]


        
if args.lr != 0.0 and args.epochs != 0 :
    lr_schedule = warmup(args.lr,warmup_epochs)+[(args.lr, args.epochs),(args.lr/2, args.epochs),(args.lr/10, args.epochs)]



print("\nlr_schedule: ", lr_schedule)

for lr,ep in lr_schedule:
    # TRAIN
    initial_epoch = model.optimizer.iterations.numpy() // num_batches

    print("LR: ",lr," epochs: ",ep,"  from:",initial_epoch, " to ",initial_epoch+ep, "\t augment = ",augment)
    # if initial_epoch == 0:
    #     reset_logs("logs")

    K.set_value(model.optimizer.learning_rate, lr)
    # data_tf = generator(train_dataset)
    history = model.fit(x = generator(train_dataset, augment = augment) ,epochs=initial_epoch+ep, steps_per_epoch = num_batches, verbose=0,
                        callbacks=[earlly_stop_cb, tensorboard_cb, checkpoint_cb], initial_epoch=initial_epoch,
                        validation_data = generator(val_dataset, augment = augment), validation_steps = val_batches)
    
    # history = model.fit(x = (x_train_new, y_train_new[:,:-1,:], emb_train_new), y = y_train_new[:,1:,:], epochs=initial_epoch+ep, initial_epoch=initial_epoch,
    #                          validation_data = ((x_valid_new, y_valid_new[:,:-1,:], emb_valid_new), y_valid_new[:,1:,:]),
    #                          callbacks=[earlly_stop_cb, tensorboard_cb, checkpoint_cb], batch_size=bs,verbose=0)


    best_index = np.argmin(history.history['val_loss'])
    val_loss = history.history['val_loss'][best_index]
    loss = history.history['loss'][best_index]

    total_epochs = model.optimizer.iterations.numpy() // num_batches
    print("BEST val_loss: ",val_loss, "\t\tloss: ", loss)
    if val_loss<min_val_loss:
        min_val_loss = val_loss
        best_model = model_file
        # print("NEW BEST MODEL: ",model_file)
    
    evaluate_model(model, epoch = total_epochs)

    # if total_epochs > warmup_epochs:
        # evaluate_model(model, epoch = total_epochs)

    # with file_writer.as_default():
    #     for ep_ in range(initial_epoch,total_epochs):
    #         tf.summary.scalar('lr', data=lr, step=ep_)



print("\n\n ----------------------------------------")
print("-----      MODEL FINAL EVALUATION       ----")
print(" ----------------------------------------")


model = load_model(model_file)
# compile(model)
evaluate_model(model, epoch = total_epochs)


print("\n\n ----------------------------------------")
print("-----           REFINING              ----")
print(" ----------------------------------------")
refine = True if args.refine == 1 else False
print("refine is :", refine)

if refine:
    ep = 1000
    lr = 5e-5
    initial_epoch = total_epochs

    print("LR: ",lr," epochs: ",ep,"  from:",initial_epoch, " to ",initial_epoch+ep,"\t augment = ",augment)
    # if initial_epoch == 0:
    #     reset_logs("logs")

    K.set_value(model.optimizer.learning_rate, lr)
    # data_tf = generator(train_dataset)


    model_name = "refined_"+model_name
    model_file = os.path.join(models_path,model_name+".h5")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=0,save_format="tf",
                                                        monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(x = generator(train_dataset,augment=False) ,epochs=initial_epoch+ep, steps_per_epoch = num_batches, verbose=0,
                        callbacks=[earlly_stop_cb, tensorboard_cb, checkpoint_cb], initial_epoch=initial_epoch,
                        validation_data = generator(val_dataset,augment=False), validation_steps = val_batches)


    new_model = load_model(model_file)
    # compile(new_model)
    evaluate_model(new_model,epoch = total_epochs)


