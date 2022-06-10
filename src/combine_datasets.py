
from motion_refiner_4D import Motion_refiner
import argparse
import os
import numpy as np
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default='/home/tum/data/data/',
                    help='Dataset directory.')


args = parser.parse_args()

traj_n = 40
mr = Motion_refiner(load_models=False,traj_n = traj_n)

set_folder = "4D_2to6_objs_norm/"

dataset_name = "4D_10000_objs_2to6_norm_"
X,Y, data = mr.load_dataset(dataset_name, filter_data = False, base_path=args.dataset_dir)

dataset_name = "4D_100000_objs_2to6_norm"
print(type(data))
for i in [0,1,3,4,6,7]:
    
    dataset_folder = args.dataset_dir+set_folder+str(i)
    print(dataset_folder+"X"+dataset_name+".npy")
    if os.path.exists(dataset_folder+"X"+dataset_name+".npy"):
        print("reading", dataset_folder+"X"+dataset_name+".npy")

        X_,Y_, data_ = mr.load_dataset(dataset_name, filter_data = False, base_path=dataset_folder)
        X = np.concatenate([X,X_])
        Y = np.concatenate([Y,Y_])
        data = data + data_

print(X.shape)
print(Y.shape)

print(len(data))

print("saving data...")

dataset_name = "4D_80000"

# # # ------- save pre processed data -------
mr.save_XY(X, Y, x_name="X"+dataset_name,y_name="Y"+dataset_name,base_path=args.dataset_dir)
mr.save_data(data,data_name="data"+dataset_name,base_path=args.dataset_dir)

