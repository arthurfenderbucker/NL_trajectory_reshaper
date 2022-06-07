
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for plotting the 3-D plot.
from data_generator.labels_generator import Label_generator
from data_generator.data_gen_utils import *
from src.motion_refiner_4D import Motion_refiner
from src.functions import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

traj_n = 40
mr = Motion_refiner(traj_n = traj_n)

images_base_path="/home/mirmi/Arthur/dataset/"
obj_lib_file= "/home/mirmi/Arthur/trajnlp_ws/src/NL_trajectory_reshaper/imagenet1000_clsidx_to_labels.txt"
dataset_name = "4D_100000_objs_2to6_norm"

# dg = data_generator({'dist':1,'speed':1, 'cartesian':1}, obj_lib_file= obj_lib_file, images_base_path=images_base_path)
# data = dg.generate(25000,4)

data = mr.load_data(data_name="data_raw"+dataset_name)
print(len(data))
# print("DONE generating")
# mr.save_data(data,data_name="data_raw"+dataset_name)
# print("raw data saved")

## ------- processed data -------
for i in range(10):
    print("starting iter: ",i)
    X,Y = mr.prepare_data(data[i*10000:(i+1)*10000],deltas=False)
    print("X: ",X.shape)
    print("Y: ",Y.shape)
    print("DONE computing embeddings")
    print("saving data...")
    # ------- save pre processed data -------
    mr.save_XY(X, Y, x_name="X"+i+dataset_name,y_name="Y"+i+dataset_name)
    mr.save_data(data,data_name="data"+i+dataset_name)
    print("DONE ",i)