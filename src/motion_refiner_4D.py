import clip
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import interpolate
import random
import re
from PIL import Image
import torch
import transformers as ppb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import warnings
try:
    from motion_refiner.src.functions import *
except:
    try:
        from src.functions import *
        from src.config import *

    except:
        from functions import *
        from config import *



warnings.filterwarnings('ignore')

MAX_NUM_OBJS = 6
MIN_NUM_OBJS = 2

base_path = "data/"


class Motion_refiner():
    def __init__(self, traj_n=40, verbose=0, load_models=True, locality_factor=True, poses_on_features=True, clip_only=False, load_precomp_emb=False):
        """
        traj_n: max num of waypoints for the interpolated trajectory
        verbose: 0 = none, 1 = prints
        """
        # super().__init__()
        self.traj_n = traj_n
        self.verbose = verbose
        self.device = "cuda"
        self.seed = 42
        self.clip_only = clip_only

        self.BERT_token_len = 19
        self.precomputed_img_emb = {}
        self.precomputed_obj_names_emb = {}


        if load_models:
            if not clip_only:
                print("loading BERT model... ",end="")
                self.BERT_model, self.BERT_tokenizer = self.load_bert()
                print("done")
            
            # self.pipe_bert,self.model_bert,self.tokenizer_bert = self.load_pipeline('distilbert-base-uncased',ppb.TFDistilBertModel)
            print("loading CLIP model... ",end="")
            self.CLIP_model, self.CLIP_preprocess = self.load_CLIP()
            print("done")

            if load_precomp_emb:
                print("loading precomputed CLIP text embbedings... ",end="")
                with open(image_dataset_folder+"obj_names_clip_emb.json") as f:
                    self.precomputed_obj_names_emb = json.load(f)
                print("done")
                print("loading precomputed CLIP img embbedings... ",end="")
                with open(image_dataset_folder+"images_clip_emb.json") as f:
                    self.precomputed_img_emb = json.load(f)
                print("done")
                # print(self.precomputed_img_emb.keys())
            print("DEVICE: ",self.device)

        self.locality_factor = locality_factor

        

        self.n_text_emb = 768
        if clip_only:
            print("clip only")
            self.n_text_emb = 512 #using clip textual embeding intead of bert

        if locality_factor:
            self.n_text_emb += 1 #appending locality factor

        self.n_objs = MAX_NUM_OBJS
        self.dim = 4
        
        self.feature_indices = np.array(range(self.n_text_emb))
        self.obj_sim_indices = np.array(
            range(self.n_text_emb, self.n_text_emb+self.n_objs))
        self.obj_poses_indices = np.array(
            range(self.n_text_emb+self.n_objs, self.n_text_emb+self.n_objs*(3+1)))
        self.traj_indices = np.array(
            range(self.n_text_emb+self.n_objs*4, self.n_text_emb+self.n_objs*(3+1)+self.traj_n*self.dim))

        self.last_index=self.n_text_emb+self.n_objs*(3+1)+self.traj_n*self.dim

        if poses_on_features:
            self.embedding_indices = np.concatenate(
                [self.feature_indices, self.obj_sim_indices, self.obj_poses_indices])
        else:
            self.embedding_indices = np.concatenate(
                [self.feature_indices, self.obj_sim_indices])

    def load_bert(self, verbose=0):
        """load a pre-trained BERT model (DistilBERT)"""

        # For DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (
            ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)
        if self.verbose > 0:
            print(" ----- BERT model loaded -----")

        return model, tokenizer

    def load_pipeline(self, name, model_name):
        model = model_name.from_pretrained(name)
        tokenizer = ppb.AutoTokenizer.from_pretrained(
            name, padding='max_length', max_length=48, do_lower_case=True, pad_to_max_length=True)
        # tokenizer = DistilBertTokenizer.from_pretrained(name, max_length=48, do_lower_case=True, pad_to_max_length=True)

        pipe = ppb.pipeline('feature-extraction',
                            model=model, tokenizer=tokenizer)
        return pipe, model, tokenizer

    def load_GPT2(self, verbose=0):

        pretrained_weights = "distilgpt2"
        model = ppb.TFGPT2LMHeadModel.from_pretrained(pretrained_weights)
        tokenizer = ppb.GPT2Tokenizer.from_pretrained(pretrained_weights)
        if self.verbose > 0:
            print(" ----- GPT2 model loaded -----")
        return model, tokenizer

    def load_CLIP(self, verbose=0):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', self.device)

        # model.cuda().eval()
        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        if self.verbose > 0:
            print(" ---- CLIP model loaded ----- ")
            print("Model parameters:",
                  f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
            print("Input resolution:", input_resolution)
            print("Context length:", context_length)
            print("Vocab size:", vocab_size)

        return model, preprocess
    def load_image(self, img_path):
            
        img = self.CLIP_preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
        return img

    def load_dataset(self, dataset_name, filter_data=False, base_path=base_path, lower_limit=-0.78, upper_limit=0.78):

        # ------- load data --------
        print("loading dataset: ",dataset_name,end=" ...")
        X_, Y_ = self.load_XY(x_name="X"+dataset_name,y_name="Y"+dataset_name, base_path=base_path)
        data_ = self.load_data(data_name="data"+dataset_name,base_path=base_path)
        feature_indices, obj_sim_indices, obj_poses_indices, traj_indices = self.get_indices()
        print("done")

        print("raw X:",X_.shape,"\tY:",Y_.shape)
        if filter_data:
            X,Y, data, i_invalid = filter(X_,Y_,data_,lower_limit=lower_limit, upper_limit=upper_limit) #for delta predictions
            print("filtered X:",X.shape,"\tY:",Y.shape)
        else:
            X,Y, data = X_,Y_, data_
        # X,Y, data, i_invalid = filter(X_,Y_abs,data_,lower_limit=0) #for wp predictions


        # print("X shape: %s\t min: %f \t max %f" % limits(X[:,traj_indices]))
        # print("Y shape: %s\t min: %f \t max %f" % limits(Y))
        if self.last_index != X.shape[-1]:
            print("ATTENTION!!! dataset and MR indexes dont match!!!")
            print("motion refiner final index = ",self.last_index)
            print("dataset final index = ",X.shape[-1])

        return X,Y, data

    def split_dataset(self, X, Y, test_size=0.2, val_size=0.1):
        reset_seed(self.seed)
        # Split the data: 70% train 20% test 10% validation
        n_samples, input_size = X.shape
        X_train_, X_test, y_train_, y_test, indices_train_, indices_test= train_test_split(X, Y,np.arange(n_samples), test_size=test_size, random_state=self.seed,shuffle= True)
        X_train, X_valid, y_train, y_valid, indices_train, indices_val = train_test_split(X_train_, y_train_, indices_train_ ,random_state=self.seed,test_size=val_size/(1-test_size), shuffle= True)
        print("Train X:",X_train.shape,"\tY:",y_train.shape)
        print("Test  X:",X_test.shape,"\tY:",y_test.shape)
        print("Val   X:",X_valid.shape,"\tY:",y_valid.shape)
        return X_train, X_test, X_valid, y_train, y_test, y_valid, indices_train, indices_test, indices_val

    def normalize_features(self, vec, axis=-1):
        row_sums = vec.sum(axis=axis)
        return vec/ row_sums[:, np.newaxis]

    def compute_clip_similarity(self, obj_names, text, images_path=None, images=None, text_feature=None):
        """computes the similarity vector between the embeded representation of a list of objects and a list of texts"""

        if not text_feature is None:
            # print("pre comp text features")
            text_clip_features = text_feature
        else:
            text_clip_features = self.get_clip_text_features([text])
        text_clip_features=text_clip_features.astype(np.float32)
        text_clip_features /=  np.linalg.norm(text_clip_features, axis=-1, keepdims=True)



        if len(self.precomputed_obj_names_emb.keys()) > 0:
            # print("pre comp obj names features")
            obj_names_features = np.array([self.precomputed_obj_names_emb[o] for o in obj_names])
        else:
            obj_names_features = self.get_clip_text_features(obj_names)
        obj_names_features = obj_names_features.astype(np.float32)
        obj_names_features /= np.linalg.norm(obj_names_features, axis=-1, keepdims=True).astype(np.float32)

        if not images_path is None: #using saved image
            if len(self.precomputed_img_emb.keys()) > 0:
                # print("pre comp image_features")
                image_features = np.array([self.precomputed_img_emb[img_path] for img_path in images_path])
            else:

                image_features = self.get_image_features(images_path, imgs=images)
            image_features = image_features.astype(np.float32)
            image_features /=  np.linalg.norm(image_features, axis=-1, keepdims=True).astype(np.float32)

            similarity_text_image = text_clip_features @ image_features.T
            similarity = similarity_text_image
        else:


            similarity_text_name = text_clip_features @ obj_names_features.T
            similarity = similarity_text_name

        return similarity


    # def compute_clip_similarity_(self, obj_names, text, images_path=None, images=None):
    #     """computes the similarity vector between the embeded representation of a list of objects and a list of texts"""

    #     token_obj_name = clip.tokenize([o for o in obj_names]).to(self.device)
    #     token_clip_text = clip.tokenize(text).to(self.device)

    #     with torch.no_grad():
    #         obj_names_features = self.CLIP_model.encode_text(
    #             token_obj_name).float()
    #         text_clip_features = self.CLIP_model.encode_text(
    #             token_clip_text).float()
    #         if not images_path is None:
                
    #             if len(self.precomputed_img_emb.keys()) > 0:
    #                 print("precomputed_img_emb")
    #                 image_features = torch.tensor([self.precomputed_img_emb[img_path] for img_path in images_path])
    #             else:
    #                 images = [self.load_image(img_path) for img_path in images_path]
    #                 image_input = torch.tensor(torch.cat(images)).to(self.device)
    #                 image_features = self.CLIP_model.encode_image(image_input).float()

    #         if not images is None:
    #             image_input = torch.tensor(torch.cat(images)).to(self.device)
    #             image_features = self.CLIP_model.encode_image(image_input).float()
                

    #     obj_names_features /= obj_names_features.norm(dim=-1, keepdim=True)
    #     text_clip_features /= text_clip_features.norm(dim=-1, keepdim=True)
        
    #     similarity_text_name = text_clip_features.cpu().numpy() @ obj_names_features.cpu().numpy().T

    #     if not images_path is None:
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         # similarity_image_name = obj_names_features.cpu().numpy() @ image_features.cpu().numpy().T #similarity between object image and object name
    #         # similarity_text_image = image_features.cpu().numpy() @ obj_names_features.cpu().numpy().T
    #         similarity_text_image = text_clip_features.cpu().numpy() @ image_features.cpu().numpy().T
    #         # similarity_text_image = text_clip_features.cpu().numpy() @ obj_names_features.cpu().numpy().T

            
    #         similarity = similarity_text_image
    #     else:
    #         similarity = similarity_text_name

    #     return similarity


    def get_clip_text_features(self, text_list, bs=100):
        '''computes the CLIP text embeddings for a list of texts'''
        def text_loader(lst, n):
            for i in range(0, len(lst), n):
                yield clip.tokenize(lst[i:i + n]).to(self.device)

        all_features = []
        with torch.no_grad():
            for t in text_loader(text_list, bs):
                features = self.CLIP_model.encode_text(t)
                all_features.append(features)
        return torch.cat(all_features).cpu().numpy()
    
    
    def get_image_features(self,image_paths, bs=100, imgs=None):
        '''computes the CLIP image embeddings for a list of image files'''
        def image_loader(lst, n):
            for i in range(0, len(lst), n):
                b = lst[i:i + n]
                classes = [c.split("/")[-2] for c in b]
                if imgs is None:
                    images = torch.concat([self.CLIP_preprocess(Image.open(im)).unsqueeze(0) for im in b])
                else:

                    images = torch.concat([self.CLIP_preprocess(Image.fromarray(im)).unsqueeze(0) for im in imgs])

                yield classes,images

        all_features = []
        with torch.no_grad():
            for classes, images in image_loader(image_paths,bs):
                features = self.CLIP_model.encode_image(images.to(self.device))
                all_features.append(features)
        return torch.cat(all_features).cpu().numpy()



    def get_bert_embeding(self, data):

        for d in data:
            d["token_text"] = self.BERT_tokenizer.encode(d["text"])
            if len(d["token_text"]) > self.BERT_token_len:
                self.BERT_token_len = len(d["token_text"])

        padded_texts = np.array(
            [d["token_text"] + [0]*(self.BERT_token_len-len(d["token_text"])) for d in data])
        # to ignore the pad values
        attention_mask = np.where(padded_texts != 0, 1, 0)

        input_ids = torch.tensor(padded_texts)
        attention_mask = torch.tensor(attention_mask)

        # print(input_ids.shape)

        with torch.no_grad():
            last_hidden_states = self.BERT_model(
                input_ids, attention_mask=attention_mask)

        # embbeded by BERT
        text_features = last_hidden_states[0][:, 0, :].numpy()
        # print(text_features.shape)

        return text_features

    def compute_embeding(self, data, pipe):
        text_features_list = []
        for d in data:
            f = pipe(d["text"])
            print(f)
            f = np.squeeze(f)[:1, :]
            print(f.shape)
            text_features_list.append(f, axis=0)
        text_features = np.concat(text_features_list)
        return text_features

    def save_data(self, data, data_name="data", base_path=base_path):
        """saves the data dict into <data_name>.json"""
        data_dict = {}
        for i, d in enumerate(data):
            data_dict[str(i)] = d
            for k, v in d.items():
                if isinstance(v, np.ndarray) or torch.is_tensor(v):
                    data_dict[str(i)][k] = v.tolist()

        with open(base_path+data_name+'.json', 'w') as f:
            json.dump(data_dict, f)

    def save_XY(self, X, Y, x_name="X", y_name="Y", base_path=base_path):
        """saves the X and Y array into <x_name>.npy and <y_name>.npy"""

        np.save(base_path+x_name+".npy", X)
        np.save(base_path+y_name+".npy", Y)
        
        

    def load_XY(self, x_name="X", y_name="Y", base_path=base_path):
        X = np.load(base_path+x_name+".npy")
        Y = np.load(base_path+y_name+".npy")
        return X, Y

    def load_data(self, data_name="data", base_path=base_path):
        data_dict = {}
        with open(base_path+data_name+".json") as f:
            data_dict = json.load(f)
        data = list(data_dict.values())

        return data
    

    def image_loader(self, im):
        """loads images as RGB cropped to the clip format"""
        sigma = [0.26862954, 0.26130258, 0.27577711]
        mean = [0.48145466, 0.4578275, 0.40821073 ]
        return ((np.swapaxes(np.swapaxes(self.load_image(im).cpu().detach().numpy()[0],0,2),1,0)*sigma+mean)*255).astype('uint8')

    def interpolate_traj(self, traj, n=None, interpolation="spline"):
        """interpolates the traj for n waypoints"""
        if n is None:
            n = self.traj_n
        print(traj.shape)        
        xp, yp, zp, velp = traj[0,:], traj[1,:], traj[2,:],  traj[3,:]

        #removes duplicated wp
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp))+ np.abs(np.diff(zp)) + np.abs(np.diff(velp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]]
        zp = np.r_[zp[okay], zp[-1]]
        velp = np.r_[velp[okay], velp[-1]]


        tck_i, u_i = interpolate.splprep([xp, yp, zp], s=0.0)
        x_i, y_i, z_i = interpolate.splev(np.linspace(0, 1, n), tck_i)

        tck,u = interpolate.splprep([np.linspace(0,1,len(velp)), velp])
        velint_x, vel_i = interpolate.splev(np.linspace(0, 1, n), tck)
        
        return x_i, y_i, z_i, vel_i

    def rotate(xy, origin, radians):
        """Use numpy to build a rotation matrix and take the dot product."""
        c, s = np.cos(radians), np.sin(radians)
        R = np.array(((c, -s), (s, c)))
        m = origin + np.dot(xy-origin, R)

    def get_indices(self):
        return self.feature_indices, self.obj_sim_indices, self.obj_poses_indices, self.traj_indices

    def prepare_data(self, data, deltas=False, label=True, interpolation="spline", verbose=1,change_img_base=None, images=None,output_forces=False, text_only=False):
        """Preprocess dataset"""

        # compute embeddings and similarity
        text_features = np.zeros((len(data),self.n_text_emb - (1 if self.locality_factor else 0)))
        
        text_list = [d["text"] for d in data]
        clip_text_features = self.get_clip_text_features(text_list)
        if self.clip_only:
            text_features = clip_text_features
        else:
            text_features = self.get_bert_embeding(data)

        print("DONE - computing textual embeddings", text_features.shape)

        for i, d in tqdm(enumerate(data)):
            if not text_only:
                image_paths = d["image_paths"]

                # d["obj_names_features"], d["text_clip_features"], d["image_features"], d['similarity'] = self.compute_clip_similarity(
                #     d["obj_names"], [d["text"]], images_path = image_paths)
                            
                if not change_img_base is None:
                    for ti in range(len(image_paths)):
                        image_paths[ti] = image_paths[ti].replace(change_img_base[0], change_img_base[1]) 
                    #     print(image_paths[ti])
                    # print(change_img_base[0], change_img_base[1])
            else:
                image_paths = None
                
            d['similarity'] = self.compute_clip_similarity(d["obj_names"], [d["text"]], images_path = image_paths, text_feature=clip_text_features[i,np.newaxis], images=images)
            # print(d["obj_names"])
            # print(d['similarity'])
            # print(self.compute_clip_similarity_(d["obj_names"], [d["text"]], images_path = image_paths, images=None))
        print("DONE - computing similarity vectors ")

        X_list = []
        Y_list = []
        # prepare data
        for i, d in tqdm(enumerate(data), disable=verbose):
            traj = np.array(d["input_traj"])
            x_i, y_i, z_i, vel_i = traj[:,0],traj[:,1],traj[:,2], traj[:,3]
            
            # interpolate the traj
            # x_i, y_i = self.interpolate_traj(
            #     np.array(d["input_traj"]).T, interpolation=interpolation)

            if label:
                traj_o = np.array(d["output_traj"] if not output_forces else d["forces"])
                x_o, y_o, z_o, vel_o = traj_o[:,0],traj_o[:,1],traj_o[:,2], traj_o[:,3]

                if deltas:
                    y = np.concatenate([x_o-x_i, y_o-y_i,z_o - z_i,vel_o-vel_i],
                                       axis=0)  # compute deltas
                else:
                    y = np.concatenate([x_o, y_o, z_o, vel_o], axis=0)  # compute deltas

                Y_list.append(y)

            sim_mask = np.zeros([len(d['similarity'][0])])
            sim_mask[np.argmax(d['similarity'][0])] = 1

            # print(i)
            # print("\n----------------------------------")
            # print(d["text"])
            # print(d["obj_names"])
            # print(d['similarity'])
            # print("poses: ", d["obj_poses"])

            # print(sim_mask)
            # print("----------------------------------")

            sim = pad_array(np.array(d['similarity'][0]),MAX_NUM_OBJS,axis=-1)
            obj_poses = pad_array(np.asarray(d["obj_poses"]),MAX_NUM_OBJS,axis=0)
            # sim = sim_mask
            if self.locality_factor:
                locality_factor= np.array([d["locality_factor"]])
                sim = np.concatenate([locality_factor,sim], axis=0)
            x = np.concatenate([sim, obj_poses.flatten(order="F"), x_i, y_i, z_i, vel_i], axis=0)
            X_list.append(x)
        print("DONE - concatenating ")

        X_ = np.stack(X_list, axis=0)
        X = np.concatenate((text_features, X_), axis=1)
        Y = None
        if label:
            Y = np.stack(Y_list, axis=0)
        return X, Y

    def prepare_x(self, x):
        objs = pad_array(list_to_wp_seq(x[:,self.obj_poses_indices],d=3),4,axis=-1)
        trajs = list_to_wp_seq(x[:, self.traj_indices],d=4)
        return np.concatenate([objs, trajs], axis=1)

    def apply_interaction(self, model, d, text,  label=False, images=None ):
        data_new = []
        data_new.append({"input_traj": d["input_traj"], "output_traj": d["output_traj"], "text": text, "obj_names": d["obj_names"],
                        "obj_poses": d["obj_poses"],"locality_factor":d["locality_factor"],"image_paths":d["image_paths"]})

        X, _ = self.prepare_data(data_new, label=label,  verbose=True, images=images)
        print(X.shape)
        traj = list_to_wp_seq(X[:, self.traj_indices],d=4)
        traj_and_obj = self.prepare_x(X)
        emb = X[:, self.embedding_indices]
        source = [traj_and_obj, None, emb]
        pred_new = generate(model, source, traj_n=self.traj_n, start_index=MAX_NUM_OBJS).numpy()
        return pred_new, traj

    def evaluate_obj_matching(self, data):
        total, score = 0, 0
        print_fail = False
        for d in data:
            # evaluate object matching
            obj = d["obj_in_text"]
            if obj != "":
                if d["obj_names"][np.argmax(d["similarity"])] == obj:
                    score += 1
                elif print_fail:
                    print("-----------FAIL---------")
                    print(d["text"])
                    print(re.split("\s|(?<!\d)[,.](?!\d)", d["text"]))
                    print(d["obj_names"][np.argmax(d["similarity"])])
                    print(d["obj_names"])
                    print(d['similarity'])
                total += 1
        print("acc: ", score/total)
        return score/total

    def follow_hard_constraints(self, traj_ref, traj_new, cont):
        traj_new_ = np.zeros_like(traj_new)
        for i, (wp_r, wp_new, cnt) in enumerate(zip(traj_ref, traj_new, cont)):
            if np.linalg.norm(wp_r-wp_new) > cnt:
                traj_new_[i, :] = wp_r - \
                    (wp_r-wp_new)*cnt/np.linalg.norm(wp_r-wp_new)
            else:
                traj_new_[i, :] = wp_new
        return traj_new_

    def addapt_to_hard_constraints(self, traj_ref, traj_new, cont):
        traj_new_ = np.zeros_like(traj_new)
        deltas_norm = np.linalg.norm(traj_ref-traj_new, axis=1)
        cnt_margin = cont - deltas_norm
        if np.min(cnt_margin) < 0:
            i = np.argmin(cnt_margin)
            alpha = cont[i]/deltas_norm[i]
            traj_new_ = traj_ref - (traj_ref-traj_new)*alpha
        else:
            traj_new_ = traj_new
        return traj_new_


if __name__ == '__main__':
    
    phi = np.linspace(0, 2.*np.pi, 40)
    r = 0.5 + np.cos(phi)         # polar coords
    x, y = r * np.cos(phi), r * np.sin(phi)   # convert to cartesian   # tck, u = splprep([x, y], s=0)
    
    
    
    n = 10
    tx = np.arange(n) + np.random.random([n])
    ty = np.arange(n)**2 + np.random.random([n])
    traj = np.concatenate(
        [tx[:, np.newaxis], ty[:, np.newaxis]], axis=1) * -0.1-0.5
    


    plt.plot(traj[:, 0], traj[:, 1])

    ni = 4
    print(traj.shape)
    # interpolates the traj
    # tck_i, u_i = interpolate.splprep(traj[:, 0], traj[:, 1])
    tck_i, u_i = interpolate.splprep([traj[:, 0], traj[:, 1]], s=0.0)
    print(tck_i)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, ni), tck_i)
    plt.plot(x_i, y_i)
    plt.show()
