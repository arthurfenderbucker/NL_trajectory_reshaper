import matplotlib.collections as mcoll
import matplotlib.path as mpath
import random
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import re
from mpl_toolkits.mplot3d import Axes3D #for plotting the 3-D plot.
from scipy import interpolate
from sklearn.utils import shuffle
try:
    from data_generator.labels_generator import Label_generator
except:
    from labels_generator import Label_generator
import os
import torch
from typing import List
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer 
import json 
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


# class Text_augmentor():
#     def __init__(self):
        
#         self.model_name = 'tuner007/pegasus_paraphrase' 
#         self.torch_device = 'cpu'
#         self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
#         self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.torch_device)
#     def augment(self, input_text: str, num_return_sequences: int, num_beams: int) -> List[str]:
#         batch = self.tokenizer.prepare_seq2seq_batch([input_text], truncation=True, padding='longest', return_tensors='pt').to(self.torch_device)
#         translated = self.model.generate(**batch, max_length = 60, num_beams = num_beams, num_return_sequences = num_return_sequences, temperature = 1.5)
#         tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens = True)
#         return tgt_text

def generate_traj(n_wp = 40, N=100,n_int= 10,margin=0.4,show=False):
    min_vel = 0.01
    
    R = (np.random.rand(N)*6).astype("int") #Randomly intializing the steps
    R_vel = (np.random.rand(N)*6).astype("int") #Randomly intializing the steps
    
    x = np.zeros(N) 
    y = np.zeros(N)
    z = np.zeros(N)
    vel = np.zeros(N)
    
    x[ R==0 ] = -1; x[ R==1 ] = 1 #assigning the axis for each variable to use
    y[ R==2 ] = -1; y[ R==3 ] = 1
    z[ R==4 ] = -1; z[ R==5 ] = 1
    vel[ R_vel==0 ] = -1; vel[ R_vel==1 ] = 1
    
    x = np.cumsum(x) #The cumsum() function is used to get cumulative sum over a DataFrame or Series axis i.e. it sums the steps across for eachaxis of the plane.
    y = np.cumsum(y)
    z = np.cumsum(z)
    vel = np.cumsum(vel)
    
    if show:
        fig = plt.figure(figsize=(8,13))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax2 = fig.add_subplot(7,1,6)
        ax.plot(x, y, z,alpha=0.5) #alpha sets the darkness of the path.
        ax.scatter(x[-1],y[-1],z[-1])

    #create spline function
    f, u = interpolate.splprep([x, y, z], s=0)
    xint, yint, zint = interpolate.splev(np.linspace(0, 1, n_int), f)
    
    tck,u = interpolate.splprep([np.linspace(0,1,len(vel)), vel])
    velint_x, velint = interpolate.splev(np.linspace(0, 1, n_int), tck)


    if show:
        ax.plot(xint, yint, zint,alpha=0.5) #alpha sets the darkness of the path.
        ax2.plot(np.linspace(0,1,len(velint)),velint)


    #create spline function
    f, u = interpolate.splprep([xint, yint, zint], s=0)
    xint, yint, zint = interpolate.splev(np.linspace(0, 1, n_wp), f)
    
    tck,u = interpolate.splprep([np.linspace(0,1,len(velint)), velint])
    velint_x,velint = interpolate.splev(np.linspace(0, 1, n_wp), tck)

    if show:
        ax.plot(xint, yint, zint,alpha=0.9) #alpha sets the darkness of the path.
        
        ax2.plot(np.linspace(0,1,len(vel)),vel)
        ax2.plot(np.linspace(0,1,len(velint)),velint,color="red")

        plt.show()
    pts = np.stack([xint,yint,zint],axis=1)
    
    velint = np.expand_dims(velint,axis=-1)

    vel_min = np.min(velint,axis = 0)
    vel_max = np.max(velint,axis = 0)
    vel_norm = np.max(np.abs(vel_max-vel_min))
    vel = ((velint-vel_min)/vel_norm)*(1-margin)+margin/2-0.5

    pts_min = np.min(pts,axis = 0)
    pts_max = np.max(pts,axis = 0)
    norm = np.max(np.abs(pts_max-pts_min))
    pts  = ((pts-pts_min)/norm)*(1-margin)+margin/2-0.5

    if show:
        ax = plt.subplot(1,1,1, projection='3d')
        ax.plot(pts[:,0],pts[:,1],pts[:,2],alpha=0.9) #alpha sets the darkness of the path.
        # ax.scatter(xint[-1],yint[-1],zint[-1])
        plt.show()

    return np.concatenate([pts,vel],axis = 1)


def apply_force(pts_raw,map_cost_f,att_points=[],locality_factor=1.0):

    init_vel = pts_raw[-1:]

    pts = pts_raw.copy()
    wps = pts[:,:3]
    init_wps = wps.copy() # first 3 dim
    
    wp_diff = wps[1:]-wps[:-1]
    init_wp_dist = np.expand_dims(np.linalg.norm(wp_diff,axis=1),-1)
    
    wp_diff_2 = (wps[2:]-wps[:-2])/2
    c = wp_diff_2-wp_diff[:-1]
    init_c_dist = np.expand_dims(np.linalg.norm(c,axis=1),axis=-1)

    # init_wp_ang_cos = np.array([np.dot(wp_dist[i-1,:]/init_wp_dist[i-1], wp_dist[i,:]/init_wp_dist[i]) for i in range(1,wp_diff.shape[0])])
    # init_wp_ang = np.arccos(init_wp_ang_cos)

    mean_dist = np.average(init_wp_dist)
    # print(mean_dist)
    k = -0.1
    k_ang = 0.5
    w = -0.02
    w_self = -0.05
    step = 1
    pts_H = []

    for i in range(1000):

        wps = pts[:,:3]
        wp_diff = wps[1:]-wps[:-1]
        wp_dist = np.expand_dims(np.linalg.norm(wp_diff,axis=1),-1)
        # print(wp_dist.shape)
        # dir_wp = wp_diff/np.expand_dims(np.linalg.norm(wp_dist,axis=1),-1)
        dir_wp = wp_diff/wp_dist

        f_wp = (wp_dist-init_wp_dist)*k*dir_wp

        null_wp = [[0,0,0]]
        F_wp_l = np.concatenate([null_wp,f_wp],axis=0)
        F_wp_r = np.concatenate([-f_wp,null_wp],axis=0)

        F_wp = F_wp_l + F_wp_r
        F_wp = np.concatenate([F_wp,np.zeros(F_wp[:,-1:].shape)],axis=-1) # add the speed component


        wp_diff_2 = (wps[2:]-wps[:-2])/2
        c = wp_diff_2-wp_diff[:-1]
        c_dist = np.expand_dims(np.linalg.norm(c,axis=1),-1)
        
        c_dir = np.divide(c, c_dist, out=np.zeros_like(c), where=c_dist!=0)
        c_delta = (c_dist-init_c_dist)

        F_ang = c_delta*k_ang*c_dir
        F_ang = np.concatenate([null_wp,F_ang,null_wp],axis=0)
        F_ang = np.concatenate([F_ang,np.zeros(F_ang[:,-1:].shape)],axis=-1) # add the speed component

        
        F_ext = np.zeros_like(pts_raw)
        
        args = {"max_r":locality_factor}
        for func in map_cost_f:

            F_ext += func(pts,args)*w
        
        self_diff = wps-init_wps
        self_dist = np.expand_dims(np.linalg.norm(self_diff,axis=1),-1)
        dir_self = np.divide(self_diff, self_dist, out=np.zeros_like(self_diff), where=self_dist!=0)
        F_self = self_dist*dir_self*w_self
        F_self = np.concatenate([F_self,np.zeros(F_self[:,-1:].shape)],axis=-1) # add the speed component


        delta = (F_ext + F_wp + F_self + F_ang)* mean_dist
        delta[0] = [0]*pts_raw.shape[-1]
        # delta[-1] = [0]*pts_raw.shape[-1]
        pts = pts + delta * step
        # if i % 30 == 0:
        #     pts_H.append(pts)
    pts_H.append(pts)
    return pts_H, F_wp





# =============== data generator ==============
MAX_NUM_OBJS = 6
MIN_NUM_OBJS = 2

def find_obj_in_text(obj_names,text):

    obj = ""
    for w in re.split("\s|(?<!\d)[,.](?!\d)",text):
        if w in obj_names:
            obj = w
    return obj


class data_generator():

    def __init__(self, change_types:dict, obj_lib_file="imagenet1000_clsidx_to_labels.txt", images_base_path = ""):
        
        self.change_types = change_types
        # self.label_generators = []
        # self.label_generators_probs
        # for ct,prob in change_types.items():

        # self.text_ag = Text_augmentor()
        self.obj_library = {}
        self.margin = 0.4
        self.images_base_path = images_base_path

        with open(obj_lib_file) as f:
            self.obj_library = json.load(f)


    def get_objs(self, num_objs:int) -> dict:

        obj_classes = random.sample(self.obj_library.keys(),num_objs)
        obj_names = [random.choice(self.obj_library[o]) for o in obj_classes]

        obj_pt = np.random.random([num_objs,3])*(1-self.margin)+self.margin/2-0.5
        objs_dict  = {}
        for x,y,z,name,c in zip(obj_pt[:,0],obj_pt[:,1],obj_pt[:,2],obj_names, obj_classes):
            objs_dict[name] = {"value":{"obj_p":[x,y,z]}, "class":c}

        return obj_names,obj_classes, obj_pt, objs_dict
    
    def generate(self, maps = 32,labels_per_map=4, plot=False,N=100, n_int=10):
        """generates maps * labels_per_map samples"""
        data = []

        for mi in tqdm(range(maps)):
            num_objs = random.randint(MIN_NUM_OBJS,MAX_NUM_OBJS)
            obj_names,obj_classes, obj_pt, objs_dict = self.get_objs(num_objs)

            lg = Label_generator(objs_dict)

            sample_done = True
            while(sample_done):
                try:
                    n_int_ = random.choice(range(n_int[0],n_int[1])) if not n_int is int else n_int
                    N_ = random.choice(range(N[0],N[1])) if not N is int else N
                    
                    pts = generate_traj(n_wp = 40, N=N_,n_int=n_int_,show=False, margin=self.margin)
                    sample_done = False
                except:
                    print("map:",mi," - error generating the map")

            lg_ct = random.choices(list(self.change_types.keys()), weights=list(self.change_types.values()))
            lg.generate_labels(lg_ct, shuffle=True)
            # print("Change type:", lg_ct, "  len = ", len(lg.labels))

            for i,(text, map_cost_f) in enumerate(lg.sample_labels(labels_per_map)):
                map_cost_f_list = [map_cost_f]
                # num_interactions = 2

                # for j in range(num_interactions-1):
                #     new_text, new_map_cost_f = lg.sample_labels(1)[0]
                #     map_cost_f_list.append(new_map_cost_f)
                #     text+=" and "+new_text

                # print("\nORIGINAL:", text)
                locality_factor = np.random.random()*0.6+0.3

                pts_new = apply_force(pts,map_cost_f_list,locality_factor=locality_factor)[0][0]
                # print("AUGMENTED:")

                obj = find_obj_in_text(obj_names,text)
                
                # if 0 and lg_ct[0]!='cartesian':
                #     text_aug = self.text_ag.augment(text,3,3)

                #     text_list = [text]

                    
                #     for ti in text_aug:
                #         if obj == "" or (obj != "" and obj in re.split("\s|(?<!\d)[,.](?!\d)",text)):
                #             # print(ti)
                #             text_list.append(ti)
                #         else:
                #             pass
                #             # print("FAIL (",obj,") : ",ti, ti.split())
                    
                #     text = random.choice(text_list) # chose final text among paraphrased possibilities 
                
                
                # print("FINAL: ", text)
                # if plot:
                #     plot_samples(text,pts,[pts_new],objs=objs, color_traj  =True)

                objs_img_base_paths = [self.images_base_path+c+"/"+n for c,n in zip(obj_classes,obj_names)]
                obj_img_paths = [im_path+"/"+random.choice(os.listdir(im_path)) for im_path in objs_img_base_paths]

                data.append({"input_traj":pts,
                            "output_traj":pts_new,
                            "text":text,
                            "obj_names":obj_names,
                            "obj_poses":obj_pt,
                            "obj_classes":obj_classes,
                            "obj_in_text":obj,
                            "change_type":lg_ct[0],
                            "map_id":mi,
                            "image_paths":obj_img_paths,
                            "locality_factor":locality_factor
                            })

        return data
        # return x,y,texts,meta


# dg = data_generator({'dist':1,'speed':1, 'cartesian':1})
# dg.next()
# dg.next()
# dg.next()




