from ast import arg
from comet_ml import Experiment
import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_musher import GPT, GPTConfig
from mingpt.model_naiveCNN import NaiveCNN
from mingpt.model_resnetdirect import ResnetDirect, ResnetDirectWithActions
from mingpt.trainer_musher import Trainer, TrainerConfig
from mingpt.trainer_resnet import TrainerResnet, TrainerResnetConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
# import blosc
import argparse
#from create_dataset import create_dataset
from datasets import ILDataset, ILVideoDataset, MushrVideoDataset
import os

import time

from torchvision import datasets, models, transforms
from torch.utils import data

from torch.utils.tensorboard import SummaryWriter

import utils.augmentation as A
import numpy as np
#from dataset import Multimodal_zip_full
from torch.utils.data.sampler import SequentialSampler

#====================================================
# Args for sure used
parser = argparse.ArgumentParser()

# general experiment params 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=20)


parser.add_argument('-j', '--workers', default=64, type=int)
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--dataset_dir', default='/home/azureuser/hackathon_data/hackathon_data',
                    help='Dataset directory.')
parser.add_argument('--outdata_dir', default='/home/azureuser/hackathon_data/aml_outputs')
parser.add_argument('--save_freq', default=1, type=int)
parser.add_argument('--prefix', default='output', type=str)
parser.add_argument('--train_ann_file_name', default='train_ann_pose_supersmall.json',
                    help='Training annotation file name (default: bc_v5_n0_train_ann.txt).')
parser.add_argument('--val_ann_file_name', default='train_ann_pose_supersmall.json',
                    help='Validation annotation file name (default: bc_v5_n0_val_ann.txt).')

# parser.add_argument('--load_gt_map', default=False, type=bool)
# parser.add_argument('--train_mode', default='e2e', type=str)
parser.add_argument('--load_gt_map', default=True, type=bool)
parser.add_argument('--train_mode', default='map', type=str)

parser.add_argument('--gt_map_file_name', default='bravern_floor.pgm', type=str)
parser.add_argument('--local_map_size_m', default=20, type=int)
parser.add_argument('--map_center', default=[-32.925, -37.3])
parser.add_argument('--map_res', default=0.05, type=float)
parser.add_argument('--map_recon_dim', default=128, type=int)

# parser.add_argument('--model_type', type=str, default='ResnetDirectWithActions') 
parser.add_argument('--model_type', type=str, default='GPT')  
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=6e-4) 
# parser.add_argument('--state_tokenizer', default='conv2D', type=str)
parser.add_argument('--state_tokenizer', default='resnet18', type=str)

parser.add_argument('--clip_len', default=16, type=int)
parser.add_argument('--restype', default='resnet18', type=str)
parser.add_argument('--use_pred_state', action='store_true', help='Use state prediction for training')
parser.add_argument('--pretrained_encoder_path', default='', type=str)
parser.add_argument('--pretrained_model_path', default='', type=str)
parser.add_argument('--loss', default='MSE', type=str)
parser.add_argument('--map_decoder', default='deconv', type=str)

parser.add_argument('--num_bins', default=5, type=int)
parser.add_argument('--rebalance_samples', default=False, type=bool)

parser.add_argument('--use_raycast_img', default=False, type=bool)
parser.add_argument('--img_dim', default=224, type=int)
parser.add_argument('--flatten_img', default=False, type=bool)

#====================================================
# Args that are probably not used 
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--num_seq', default=8, type=int)
parser.add_argument('--pred_step', default=3, type=int)
parser.add_argument('--flip', default=False, type=bool)
parser.add_argument('--zip_file_name', type=str, default='')
#====================================================

args = parser.parse_args()

set_seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
device = torch.device('cuda')
# device = torch.device('cpu')

# choose state tokenizer
# TODO: create more types of augmentations like flip LR (remember to flip action and GT map as well), or add noise?

if args.state_tokenizer == 'conv2D':
    state_tokenizer = 'conv2D'
    train_transform = transforms.Compose([
    transforms.Resize(args.img_dim),  # Only resize.
    transforms.ToTensor()
    ])  
elif args.state_tokenizer == 'resnet18':
    state_tokenizer = 'resnet18'
    train_transform = transforms.Compose([
    transforms.ToTensor()
    ]) 
elif args.state_tokenizer == 'resnet50':
    state_tokenizer = 'resnet50'
    train_transform = transforms.Compose([
    transforms.ToTensor()
    ]) 
elif args.state_tokenizer == 'compass':
    state_tokenizer = 'compass'
    train_transform = transforms.Compose([
    transforms.Resize(args.img_dim),  # Only resize.
    transforms.ToTensor()
    ]) 
elif args.state_tokenizer == 'FCL':
    state_tokenizer = 'FCL'
else:
    print('Not supported!')


# datasets depending on the model type

train_dataset = MushrVideoDataset(
    dataset_dir=args.dataset_dir, ann_file_name=args.train_ann_file_name, 
    transform=train_transform, gt_map_file_name=args.gt_map_file_name, local_map_size_m=args.local_map_size_m,
    map_center=args.map_center, map_res=args.map_res, use_raycast_img=args.use_raycast_img, clip_len=args.clip_len,
    flatten_img=args.flatten_img, load_gt_map=args.load_gt_map, rebalance_samples=args.rebalance_samples, num_bins=args.num_bins, map_recon_dim=args.map_recon_dim)
val_dataset = MushrVideoDataset(
    dataset_dir=args.dataset_dir, ann_file_name=args.val_ann_file_name, 
    transform=train_transform, gt_map_file_name=args.gt_map_file_name, local_map_size_m=args.local_map_size_m,
    map_center=args.map_center, map_res=args.map_res, use_raycast_img=args.use_raycast_img, clip_len=args.clip_len,
    flatten_img=args.flatten_img, load_gt_map=args.load_gt_map, rebalance_samples=args.rebalance_samples, num_bins=args.num_bins, map_recon_dim=args.map_recon_dim)

print(f'Training set: {len(train_dataset)}, val set: {len(val_dataset)}.')

#=========================train val split======================================

#dataset_size = len(train_dataset)
#indices = list(range(dataset_size))
#split = int(np.floor(0.2 * dataset_size))
#train_indices, val_indices = indices[split:], indices[:split]
#
## Creating PT data samplers and loaders:
#train_sampler = SequentialSampler(train_indices)
#valid_sampler = SequentialSampler(val_indices)

# ======================== random split ======================================
#model_dataset = ILVideoDataset(
#    dataset_dir=args.dataset_dir, ann_file_name=args.train_ann_file_name, zip_file_name=args.zip_file_name, 
#    transform=train_transform, clip_len=args.clip_len, use_flow=(args.flow_type != ''), use_flow_vis_as_img=args.use_flow_vis_as_img,
#    use_depth_vis_as_img=args.use_depth_vis_as_img, flatten_img=args.flatten_img)
#dataset_size = len(model_dataset)
#train_count = int(0.8 * dataset_size)
#valid_count = dataset_size - train_count
#
#train_dataset, val_dataset = torch.utils.data.random_split(
#    model_dataset, (train_count, valid_count)
#)


#=========================train val split======================================

#dataset_size = len(train_dataset)
#indices = list(range(dataset_size))
#split = int(np.floor(0.2 * dataset_size))
#train_indices, val_indices = indices[split:], indices[:split]
#
## Creating PT data samplers and loaders:
#train_sampler = SequentialSampler(train_indices)
#valid_sampler = SequentialSampler(val_indices)


train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = train_loader
test_dataset = val_loader
vocab_size = 100
block_size = args.clip_len * 2
max_timestep = 7

def set_path(args):
    import datetime
    today = datetime.date.today()
    t_now = time.time()
    exp_path = '{args.outdata_dir}/log_{args.prefix}/{args.model_type}'.format(args=args)
    exp_path = os.path.join(args.outdata_dir, 'log_'+args.prefix, os.environ.get('AMLT_EXPERIMENT_NAME',str(t_now)), 
                            args.model_type+os.environ.get('AMLT_JOB_NAME','job_name')+ '_' + str(today) + '_' + str(time.time()))
    exp_path = exp_path + '_' + str(today) + '_' + str(time.time())
    print('******* Model saved in %s **********:' %exp_path)

    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(model_path): 
        os.makedirs(model_path)

    
    tensorboard_path = os.path.join(args.outdata_dir, 'log_tensorboard', os.environ.get('AMLT_EXPERIMENT_NAME',str(t_now)), os.environ.get('AMLT_JOB_NAME','job_name'))
    if not os.path.exists(tensorboard_path): 
        os.makedirs(tensorboard_path)

    return model_path, tensorboard_path

model_path, tensorboard_path = set_path(args)

# ============== init model ======================
if args.model_type == 'GPT':
    mconf = GPTConfig(vocab_size, block_size, max_timestep,
                      n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, use_pred_state=args.use_pred_state,
                      state_tokenizer=state_tokenizer, pretrained_encoder_path=args.pretrained_encoder_path, 
                      loss=args.loss, train_mode=args.train_mode, pretrained_model_path=args.pretrained_model_path,
                      map_decoder=args.map_decoder, map_recon_dim=args.map_recon_dim)
    model = GPT(mconf, device)
elif args.model_type == 'naivecnn':
    model = NaiveCNN(args, device)
elif args.model_type == 'ResnetDirect':
    model = ResnetDirect(device, clip_len=args.clip_len, restype=args.restype)
elif args.model_type == 'ResnetDirectWithActions':
    model = ResnetDirectWithActions(device, clip_len=args.clip_len, restype=args.restype)

model.to(device)
model=nn.DataParallel(model)

# initialize a trainer instance and kick off training
epochs = args.epochs
if args.model_type == 'GPT':
    tconf = TrainerConfig(max_timestep, max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                          num_workers=args.workers, seed=args.seed, model_type=args.model_type, model_path=model_path,
                          save_freq=args.save_freq, flatten_img=args.flatten_img, loss=args.loss, train_mode=args.train_mode,
                          map_decoder=args.map_decoder, map_recon_dim=args.map_recon_dim)
    trainer = Trainer(model, train_dataset, device, test_dataset, tconf)
elif args.model_type == 'naivecnn' or args.model_type == 'ResnetDirect' or args.model_type == 'ResnetDirectWithActions':
    tconf = TrainerResnetConfig(max_timestep, max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr,
                          lr_decay=False, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                          num_workers=args.workers, seed=args.seed, model_type=args.model_type, model_path=model_path,
                          save_freq=args.save_freq, flatten_img=args.flatten_img, loss=args.loss)
    trainer = TrainerResnet(model, train_dataset, device, test_dataset, tconf)

# experiment = Experiment(
#     api_key="AYCYZUofXkf8XE3bxIaNl74Du",
#     project_name="generall",
#     workspace="rbonatti",
# )

SW = SummaryWriter(tensorboard_path, flush_secs=2)
trainer.train(experiment=None, SW=SW)

