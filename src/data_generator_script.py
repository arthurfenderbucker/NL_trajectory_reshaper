
from email.mime import base
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #for plotting the 3-D plot.
from data_generator.labels_generator import Label_generator
from data_generator.data_gen_utils import *
from motion_refiner_4D import Motion_refiner
from functions import *
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import gc
import argparse
import datetime
import threading


parser = argparse.ArgumentParser()

parser.add_argument('--traj_n', type=int, default=40)
parser.add_argument('--exp_name', default="test_script"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
parser.add_argument('--dataset_dir', default='/home/tum/data/data/')
parser.add_argument('--image_dataset_dir', default='/home/tum/data/image_dataset/')
parser.add_argument('--prefix', default='0',)
parser.add_argument('--labels_per_map',type=int, default=1)
parser.add_argument('--n_map',type=int, default=5000)
parser.add_argument('--threads',type=int, default=20)





args = parser.parse_args()



traj_n = args.traj_n
mr = Motion_refiner(traj_n = traj_n)


dataset_name = "4D_2to6_norm"
images_base_path=args.image_dataset_dir

obj_lib_file= images_base_path+"imagenet1000_clsidx_to_labels.txt"

data_folder = args.dataset_dir


folder = args.exp_name+"/"
exp_folder = os.path.join(args.dataset_dir,folder)

if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

## ------- processed data -------

def generation_thread(i):

    folder_and_prefix = folder+args.prefix+"_"+str(i)
    print(data_folder+folder_and_prefix+"data_raw"+dataset_name+".json")

    # if not os.path.exists(data_folder+folder+str(i)+"data_raw"+dataset_name+".json"):

    print(i,": starting")
    dg = data_generator({'dist':1,'speed':1, 'cartesian':1}, obj_lib_file= obj_lib_file, images_base_path=images_base_path)
    data = dg.generate(args.n_map,args.labels_per_map,N=[50,100],n_int=[3,15])
    print(i,len(data))
    print(i,"DONE generating")
    # mr.save_data(data,data_name=folder_and_prefix+"data_raw"+dataset_name, base_path=data_folder)
    # else:
    #     if os.path.exists(data_folder+folder+str(i)+"X"+dataset_name+".npy"):
    #         continue
    #     print("loading data") 
    #     data = mr.load_data(data_name=folder+str(i)+"data_raw"+dataset_name,base_path=data_folder)

    try:
        X,Y = mr.prepare_data(data,deltas=False)
        print(i,"X: ",X.shape)
        print(i,"Y: ",Y.shape)
        print(i,"DONE computing embeddings")
        print(i,"saving data...")
        # # ------- save pre processed data -------
        mr.save_XY(X, Y, x_name=folder_and_prefix+"X"+dataset_name,y_name=folder_and_prefix+"Y"+dataset_name, base_path=data_folder)
        mr.save_data(data,data_name=folder_and_prefix+"data"+dataset_name, base_path=data_folder)
        print(i,"DONE ")
    except:
        print("\n",i," !!!!!! failed computing embeddings !!!!!!\n")
    gc.collect()
    del(data)
    del(X)
    del(Y)

    # print("\n validating")
    # X_,Y_, data_ = mr.load_dataset(dataset_name, filter_data = True, base_path=args.dataset_dir)


for i in range(0,args.threads):
    try:#
        x = threading.Thread(target=generation_thread, args=(i,))
        # logging.info("starting")
        x.start()
    except:
        print("Error: unable to start thread: ",i)

