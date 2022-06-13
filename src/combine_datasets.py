
from motion_refiner_4D import Motion_refiner
import argparse
import os
import numpy as np
import random
import re
from tqdm import tqdm
import torch
import gc

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default='/home/tum/data/data/',
                    help='Dataset directory.')

parser.add_argument('--augment_text', default="None")
parser.add_argument('--new_dataset_name', default="4D_1M_scaping_factor")
parser.add_argument('--set_name', default="ditributed")
parser.add_argument('--dataset_name', default="4D_2to6_norm")



args = parser.parse_args()




if args.augment_text != "None":
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer 

    
    class Text_augmentor():
        def __init__(self):
            
            self.model_name = 'tuner007/pegasus_paraphrase' 
            self.torch_device = 'cuda'
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.torch_device)
        def augment(self, input_text: list, num_return_sequences: int, num_beams: int):
            batch = self.tokenizer.prepare_seq2seq_batch(input_text, truncation=True, padding='longest', return_tensors='pt').to(self.torch_device)
            translated = self.model.generate(**batch, max_length = 60, num_beams = num_beams, num_return_sequences = num_return_sequences, temperature = 1.5)
            tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens = True)
            return tgt_text

    text_ag = Text_augmentor()


def augment_text(text_ag, data_raw):
    text_list = [d["text"] for d in data_raw]
    
    n_factor = 3
    n_samples = 30
    new_text_list = []
    count=0
    while count < len(text_list):

        lower_i = count
        higher_i = min(count+n_samples,len(text_list))


        text_all = np.array(text_ag.augment(text_list[lower_i:higher_i],n_factor,10)).reshape(-1,n_factor)
        new_text_list = new_text_list + [random.choice(ts) for i,ts in enumerate(text_all)]

        print(higher_i)
        count+=n_samples
        for i in range(lower_i,higher_i):
            data_raw[i]["text"] = new_text_list[i]
        gc.collect()
        torch.cuda.empty_cache()
    
    return data_raw




traj_n = 40
mr = Motion_refiner(load_models=(args.augment_text!="None"),traj_n = traj_n)

set_folder = args.set_name

files_prefix = [ f.split("data")[0] for f in os.listdir(os.path.join(args.dataset_dir,set_folder)) if f[-4:]=="json"]

dataset_name = args.dataset_name

print(files_prefix)


for i,pref in enumerate(files_prefix[:20]):
    
    dataset_folder = args.dataset_dir+set_folder+str(pref)
    print(dataset_folder+"X"+dataset_name+".npy")
    if os.path.exists(dataset_folder+"X"+dataset_name+".npy"):
        print("reading", dataset_folder+"X"+dataset_name+".npy")


        X_,Y_, data_ = mr.load_dataset(dataset_name, filter_data = False, base_path=dataset_folder)


        if args.augment_text != "None":
            print(data_[0]["text"])
            data_ = augment_text(text_ag, data_)
            print(data_[0]["text"])

            X_,Y_ = mr.prepare_data(data_,deltas=False,change_img_base=["/mnt/tumdata/image_dataset/","/home/tum/data/image_dataset/"])
            print(pref,"DONE computing new embeddings")
            print(pref,"saving new data...")
            # # ------- save pre processed data -------
            mr.save_XY(X_, Y_, x_name=args.augment_text+"X"+dataset_name,y_name=+args.augment_text+"Y"+dataset_name,base_path=dataset_folder)
            mr.save_data(data_,data_name=args.augment_text+"data"+dataset_name,base_path=dataset_folder)
        
        if i ==0:
            X = X_
            Y = Y_
            data = data_
        else:
            X = np.concatenate([X,X_])
            Y = np.concatenate([Y,Y_])
            data = data + data_


print(X.shape)
print(Y.shape)
print(len(data))

print("saving data...")

dataset_name = args.new_dataset_name
while os.path.exists(dataset_folder+"X"+dataset_name+".npy"):
    print("dataset_already exists")
    dataset_name+="_"
# # # ------- save pre processed data -------
mr.save_XY(X, Y, x_name="X"+dataset_name,y_name="Y"+dataset_name,base_path=args.dataset_dir)
mr.save_data(data,data_name="data"+dataset_name,base_path=args.dataset_dir)

