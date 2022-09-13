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

from motion_refiner_4D import Motion_refiner
# from mlflow.tracking import MlflowClient
# import mlflow
from config import *
import time


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=0)
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--dataset_dir', default=data_folder)
parser.add_argument('--models_path', default=models_folder)
parser.add_argument('--dataset_name', default=dataset_name)

parser.add_argument('--exp_name', default="experimet_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

parser.add_argument('--lr', type=float, default=0.0) #use default lr decay
parser.add_argument('--num_enc', type=int, default=1)
parser.add_argument('--num_dec', type=int, default=5)
parser.add_argument('--num_dense', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--model_depth', type=int, default=400)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--dff', type=int, default=512)
parser.add_argument('--dense_n', type=int, default=512)
parser.add_argument('--concat_emb', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=10000)
parser.add_argument('--augment', type=int, default=0)
parser.add_argument('--base_model', type=str, default="None")
parser.add_argument('--refine', type=int, default=0)
parser.add_argument('--optimizer', type=str, default="adam")
parser.add_argument('--activation', type=str, default="tanh")
parser.add_argument('--ignore_features', type=int, default=0)
parser.add_argument('--norm_layer', type=int, default=1)
parser.add_argument('--num_emb_vec', type=int, default=16)
parser.add_argument('--sf', type=float, default=1.0)
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--lr_decay_patience', type=int, default=10)
parser.add_argument('--loss', default="mse")



parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--clip_only', type=bool, default=False)


args = parser.parse_args()


if args.test:
    print("-------------------------------------------")
    print("--    testing training precedure mode    --")
    print("-------------------------------------------")


concat_emb = False if args.concat_emb == 0 else True 
augment = False if args.augment == 0 else True 
ignore_features = False if args.ignore_features == 0 else True
norm_layer = False if args.norm_layer == 0 else True

delimiter ="-"


traj_n = 40
mr = Motion_refiner(load_models=False ,traj_n = traj_n, clip_only=args.clip_only)
feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = mr.get_indices()
embedding_indices = np.concatenate([feature_indices,obj_sim_indices, obj_poses_indices])


# dataset_name = "4D_10000_objs_2to6_norm_"
# dataset_name = "4D_80000"
dataset_name = args.dataset_name

print("Loading dataset: ", dataset_name)
X,Y, data = mr.load_dataset(dataset_name, filter_data = True, base_path=args.dataset_dir)



if ignore_features:
    X[:,feature_indices] = 0
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!         IGNORING FEATURES          !!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val = mr.split_dataset(X, Y, test_size=0.2, val_size=0.1)


sf = args.sf
if args.test:
    sf = 0.1

X_train = X_train[:int(X_train.shape[0]*sf),:]
y_train = y_train[:int(y_train.shape[0]*sf),:]
indices_train = indices_train[:int(indices_train.shape[0]*sf)]


X_valid = X_valid[:int(X_valid.shape[0]*sf),:]
y_valid = y_valid[:int(y_valid.shape[0]*sf),:]
indices_val = indices_val[:int(indices_val.shape[0]*sf)]

if args.test:
    X_test = X_test[:int(X_test.shape[0]*sf),:]
    y_test = y_test[:int(y_test.shape[0]*sf),:]
    indices_test = indices_test[:int(indices_test.shape[0]*sf)]


print("\nDataset size used for training !!")
print("Train X:",X_train.shape,"\tY:",y_train.shape)
print("Val   X:",X_valid.shape,"\tY:",y_valid.shape)
print("Test  X:",X_test.shape,"\tY:",y_test.shape)


print(X_valid[:3,:5])
print(y_valid[:3,:5])

#------------------------------------------------------------------------

# print(reorganize_input(np.expand_dims(np.array(range(20)),axis=0)))

# Y_abs = Y_+ X_[:,traj_indices]

# print("filtered cartesian changes X:",X.shape,"\tY:",Y.shape)


# X,Y, data, i_invalid = filter(X_,Y_abs,data_,lower_limit=0) #for wp predictions

print(limits(X[:,traj_indices]))
print(limits(Y))


#------------------------------------------------------------------------


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
from keras import backend as K
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print("\n\n devices: ",get_available_devices()) 


print("\n\nX:",X.shape,"\tY:",Y.shape)
# print("filtered: ", len(i_invalid))

n_samples, input_size = X.shape
#------------------------------------------------------------------------


# Create a new experiment if one doesn't already exist
# mlflow.create_experiment(args.exp_name)

from TF4D_decoder_only import *


reset_seed(seed)

for i in range(5):
    print(tf.random.uniform(shape=[2]))


embedding_indices = np.concatenate([feature_indices,obj_sim_indices, obj_poses_indices])
# embedding_indices = np.concatenate([feature_indices,obj_sim_indices])

features_n = len(embedding_indices)

#------------------------------------------------------------------------
#------------------------------------------------------------------------

bs=args.bs
def prepare_x(x):
  objs = pad_array(list_to_wp_seq(x[:,obj_poses_indices],d=3),4,axis=-1) # no speed
  trajs = list_to_wp_seq(x[:,traj_indices],d=4)
  #   return np.concatenate([objs,trajs],axis = 1)
  return trajs[:,:-1,:]


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
                num_heads = args.num_heads,
                dropout_rate = args.dropout,
                wp_d=4,
                num_emb_vec=args.num_emb_vec,
                bs=args.bs,
                dense_n=args.dense_n,
                num_dense=args.num_dense,
                concat_emb=concat_emb,
                features_n=features_n,
                optimizer=args.optimizer,
                norm_layer=norm_layer,
                activation=args.activation,
                loss=args.loss)

param_s = json.dumps(param)
model_name = "TF"
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

        model = load_model(args.base_model,delimiter =delimiter)
        base_model_name = args.base_model.split("/")[-1][:-3]
    else:
        base_model_name = args.base_model
        base_model_file = os.path.join(models_path,base_model_name+".h5")
        model = load_model(base_model_file,delimiter =delimiter)


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

print(model.summary())

logdir = os.path.join(models_path,"logs",model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(models_path):
    os.makedirs(models_path)



file_writer = tf.summary.create_file_writer(logdir + "/metrics")

def evaluate_model(model, epoch):

    print("epoch:",epoch)
    # print("\nwith data augmentation:")
    # result_eval_aug = model.evaluate(generator(test_dataset,stop=True))[0]

    x_test_new, y_test_new = prepare_x(X_test), list_to_wp_seq(y_test,d=4)
    emb_test_new = X_test[:,embedding_indices]

    result_eval = model.evaluate((x_test_new, y_test_new[:,:-1,:], emb_test_new), y_test_new[:,1:,:])

    pred = model.predict([x_test_new, y_test_new[:,:-1,:], emb_test_new], verbose=0)

    # print("\n ----------------------------------------")
    # print("withdata generation:")
    # test_dataset = tf.data.Dataset.from_tensor_slices((prepare_x(X_test),
    #                                                 list_to_wp_seq(y_test,d=4),
    #                                                 X_test[:,embedding_indices])).batch(X_test.shape[0])

    # g = generator(test_dataset,stop=True,augment=False)
    # x_t, y_t = next(g)
    # pred = generate(model ,x_t, traj_n=traj_n).numpy()
    # print(pred.shape)
    # result_gen = np.average((y_t - pred[:,1:,:])**2)
    # print("Test loss w generation: ",result_gen)

    # print("computing metrics...")
    metrics, metrics_h = compute_metrics(pred[:,:,:3],y_test_new[:,1:,:3])

    with file_writer.as_default():
        # tf.summary.scalar('test_result_gen', data=result_gen, step=epoch)
        tf.summary.scalar('test_result_eval', data=result_eval, step=epoch)

        for k in metrics.keys():
            tf.summary.scalar(k, data=metrics[k], step=epoch)




if args.base_model != "None":

    print("\nEvaluating base model...")
    total_epochs = model.optimizer.iterations.numpy() // num_batches
    evaluate_model(model, epoch = total_epochs)

print("\n\n ----------------------------------------")
print("-----           TRAINING             ----")
print(" ----------------------------------------")
print("starting: ",model_name )



earlly_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',  mode='min', verbose=2, patience=30)
tensorboard_cb = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=0,
                                                    monitor='val_loss', mode='min', save_best_only=True)
rlrp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_delta=1E-7)


# monitor the learning rate
class LearningRateMonitor(tf.keras.callbacks.Callback):
	# start of training
	def on_train_begin(self, logs={}):
		self.lrates = list()
 
	# end of each training epoch
	def on_epoch_end(self, epoch, logs={}):
		# get and store the learning rate
		optimizer = self.model.optimizer
		lrate = float(K.get_value(self.model.optimizer.lr))

		with file_writer.as_default():
			tf.summary.scalar('learning rate', data=lrate, step=epoch)

		self.lrates.append(lrate)

lrm = LearningRateMonitor()

# absl.logging.set_verbosity(absl.logging.ERROR)  #prevent checkpoint warnings while training



total_epochs = 0


def warmup(v,ep):
    return [(i,1) for i in np.linspace(v/ep, v, num=ep)]

warmup_epochs = 15
lr_schedule = warmup(1e-4,warmup_epochs)+[(1e-4,500)]
if args.test:
    lr_schedule = [(1e-4,1 if args.epochs == 0 else args.epochs)]
    warmup_epochs = 0

# if args.lr != 0.0 and args.epochs != 0 :
#     lr_schedule = warmup(args.lr,warmup_epochs)+[(args.lr, args.epochs)]


print("\nlr_schedule: ", lr_schedule)



t0 = time.time()
for lr,ep in lr_schedule:
    # TRAIN
    initial_epoch = model.optimizer.iterations.numpy() // num_batches

    print("LR: ",lr," epochs: ",ep,"  from:",initial_epoch, " to ",initial_epoch+ep, "\t augment = ",augment)
    # if initial_epoch == 0:
    #     reset_logs("logs")

    K.set_value(model.optimizer.learning_rate, lr)

    verbose = 0 if not args.test else 1
    history = model.fit(x = generator(train_dataset, augment = augment) ,epochs=initial_epoch+ep, steps_per_epoch = num_batches, verbose=verbose,
                        callbacks=[earlly_stop_cb, tensorboard_cb, checkpoint_cb, lrm, rlrp], initial_epoch=initial_epoch,
                        validation_data = generator(val_dataset, augment = augment), validation_steps = val_batches, shuffle=False, use_multiprocessing=False)
    
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
    
    if total_epochs > warmup_epochs:
        evaluate_model(model, epoch = total_epochs)

    # with file_writer.as_default():
    #     for ep_ in range(initial_epoch,total_epochs):
    #         tf.summary.scalar('lr', data=lr, step=ep_)


print("--- %s seconds ---" % (time.time() - t0))
print("\n\n ----------------------------------------")
print("-----      MODEL FINAL EVALUATION       ----")
print(" ----------------------------------------")


model = load_model(model_file,delimiter =delimiter)
# compile(model)
evaluate_model(model, epoch = total_epochs+1)


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

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_file, verbose=0,
                                                        monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(x = generator(train_dataset,augment=False) ,epochs=initial_epoch+ep, steps_per_epoch = num_batches, verbose=0,
                        callbacks=[earlly_stop_cb, tensorboard_cb, checkpoint_cb], initial_epoch=initial_epoch,
                        validation_data = generator(val_dataset,augment=False), validation_steps = val_batches, shuffle=False, use_multiprocessing=False)


    new_model = load_model(model_file,delimiter =delimiter)
    # compile(new_model)
    evaluate_model(new_model,epoch = total_epochs)


