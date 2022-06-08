
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for plotting the 3-D plot.
from data_generator.labels_generator import Label_generator
from data_generator.data_gen_utils import *
from src.motion_refiner_4D import Motion_refiner
from src.functions import *
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import gc

traj_n = 40
mr = Motion_refiner(traj_n = traj_n)

images_base_path="/home/mirmi/Arthur/dataset/"
obj_lib_file= "/home/mirmi/Arthur/trajnlp_ws/src/NL_trajectory_reshaper/imagenet1000_clsidx_to_labels.txt"
dataset_name = "4D_100000_objs_2to6_norm"
data_folder = "/home/mirmi/Arthur/trajnlp_ws/src/NL_trajectory_reshaper/data/"

# data = mr.load_data(data_name="data_raw"+dataset_name)
# print(len(data))
# print("DONE generating")
# mr.save_data(data,data_name="data_raw"+dataset_name)
# print("raw data saved")


bs = 10000
folder = "4D_2to6_objs_norm/"
## ------- processed data -------
for i in range(0,10):
    print(data_folder+folder+str(i)+"data_raw"+dataset_name+".json")
    if not os.path.exists(data_folder+folder+str(i)+"data_raw"+dataset_name+".json"):
        print("starting iter: ",i)
        dg = data_generator({'dist':1,'speed':1, 'cartesian':1}, obj_lib_file= obj_lib_file, images_base_path=images_base_path)
        data = dg.generate(2500,4,N=[50,100],n_int=[5,15])
        print(len(data))
        print("DONE generating")
        mr.save_data(data,data_name=folder+str(i)+"data_raw"+dataset_name)
    else:
        if os.path.exists(data_folder+folder+str(i)+"X"+dataset_name+".npy"):
            continue
        print("loading data") 
        data = mr.load_data(data_name=folder+str(i)+"data_raw"+dataset_name)

    try:
        X,Y = mr.prepare_data(data,deltas=False)
        print("X: ",X.shape)
        print("Y: ",Y.shape)
        print("DONE computing embeddings")
        print("saving data...")
        # # ------- save pre processed data -------
        mr.save_XY(X, Y, x_name=folder+str(i)+"X"+dataset_name,y_name=folder+str(i)+"Y"+dataset_name)
        mr.save_data(data,data_name=folder+str(i)+"data"+dataset_name)
        print("DONE ",i)
    except:
        print("\n!!!!!! failed computing embeddings !!!!!!\n")
    gc.collect()
    del(data)
    del(X)
    del(Y)

