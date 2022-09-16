import clip
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import interpolate
import random
import re

import torch
import transformers as ppb

import warnings
try:
    from motion_refiner.src.functions import *
except:
    try:
        from src.functions import *
    except:
        from functions import *


warnings.filterwarnings('ignore')


base_path = "data/"


class Motion_refiner():
    def __init__(self, traj_n=10, verbose=0, load_models=True):
        """
        traj_n: num of waypoints for the interpolated trajectory
        verbose: 0 = none, 1 = prints
        """
        # super().__init__()
        self.traj_n = traj_n
        self.verbose = verbose
        self.device = "cuda"

        self.BERT_token_len = 19
        if load_models:
            self.BERT_model, self.BERT_tokenizer = self.load_bert()
            # self.pipe_bert,self.model_bert,self.tokenizer_bert = self.load_pipeline('distilbert-base-uncased',ppb.TFDistilBertModel)

            self.CLIP_model, self.CLIP_tokenizer = self.load_CLIP()

        self.bert_n = 768
        self.n_objs = 3
        self.traj_n = 10
        self.feature_indices = np.array(range(self.bert_n))
        self.obj_sim_indices = np.array(
            range(self.bert_n, self.bert_n+self.n_objs))
        self.obj_poses_indices = np.array(
            range(self.bert_n+self.n_objs, self.bert_n+self.n_objs*3))
        self.traj_indices = np.array(
            range(self.bert_n+self.n_objs*3, self.bert_n+self.n_objs*3+self.traj_n*2))

        self.embedding_indices = np.concatenate(
            [self.feature_indices, self.obj_sim_indices, self.obj_poses_indices])

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

    def compute_clip_similarity(self, obj_names, text):
        """computes the similarity vector between the embeded representation of a list of objects and a list of texts"""

        token_obj_name = clip.tokenize([o for o in obj_names]).to(self.device)
        token_clip_text = clip.tokenize(text).to(self.device)

        with torch.no_grad():
            obj_names_features = self.CLIP_model.encode_text(
                token_obj_name).float()
            text_clip_features = self.CLIP_model.encode_text(
                token_clip_text).float()

        obj_names_features /= obj_names_features.norm(dim=-1, keepdim=True)
        text_clip_features /= text_clip_features.norm(dim=-1, keepdim=True)
        similarity = text_clip_features.cpu().numpy() @ obj_names_features.cpu().numpy().T

        return token_obj_name, token_clip_text, similarity

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

    def compute_bert_embeding(self, data):

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
        # TODO change data values to np arrays and tensors
        return data

    def interpolate_traj(self, traj, n=None, interpolation="spline"):
        """interpolates the traj for n waypoints"""
        if n is None:
            n = self.traj_n
        # print(traj.shape)        
        xp, yp = traj[0,:], traj[1,:]

        #removes duplicated wp
        okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
        xp = np.r_[xp[okay], xp[-1]]
        yp = np.r_[yp[okay], yp[-1]]

        tck_i, u_i = interpolate.splprep([xp, yp], s=0.0)
        x_i, y_i = interpolate.splev(np.linspace(0, 1, n), tck_i)
        return x_i, y_i

    def rotate(xy, origin, radians):
        """Use numpy to build a rotation matrix and take the dot product."""
        c, s = np.cos(radians), np.sin(radians)
        R = np.array(((c, -s), (s, c)))
        m = origin + np.dot(xy-origin, R)

    def get_indices(self):
        return self.feature_indices, self.obj_sim_indices, self.obj_poses_indices, self.traj_indices

    def prepare_data(self, data, deltas=False, label=True, interpolation="spline"):
        """Preprocess dataset"""

        # compute embeddings and similarity
        text_features = self.compute_bert_embeding(data)
        # text_features = self.compute_embeding(data,self.pipe_bert)
        for i, d in enumerate(data):
            d["token_obj_name"], d["token_clip_text"], d['similarity'] = self.compute_clip_similarity(
                d["obj_names"], [d["text"]])

        X_list = []
        Y_list = []
        # prepare data
        for i, d in enumerate(data):
            traj = np.array(d["input_traj"])

            # interpolate the traj
            x_i, y_i = self.interpolate_traj(
                np.array(d["input_traj"]).T, interpolation=interpolation)

            if label:
                x_o, y_o = self.interpolate_traj(
                    np.array(d["output_traj"]).T, interpolation=interpolation)

                # if rotate:
                #     rotate(, origin,radians)

                # y = np.concatenate([x_o-x_i,y_o-y_i],axis = 0) #compute deltas
                if deltas:
                    y = np.concatenate([x_o-x_i, y_o-y_i],
                                       axis=0)  # compute deltas
                else:
                    y = np.concatenate([x_o, y_o], axis=0)  # compute deltas

                Y_list.append(y)

            sim_mask = np.zeros([len(d['similarity'][0])])
            sim_mask[np.argmax(d['similarity'][0])] = 1

            # print("\n----------------------------------")
            # print(d["text"])
            # print(d["obj_names"])
            # print(d['similarity'])
            # print("poses: ", d["obj_poses"])

            # print(sim_mask)
            # print("----------------------------------")

            sim = np.array(d['similarity'][0])
            # sim = sim_mask
            x = np.concatenate(
                [sim, np.array(d["obj_poses"]).flatten(), x_i, y_i], axis=0)
            X_list.append(x)

        X_ = np.stack(X_list, axis=0)
        X = np.concatenate((text_features, X_), axis=1)
        Y = None
        if label:
            Y = np.stack(Y_list, axis=0)
        return X, Y

    def prepare_x(self, x):
        objs = list_to_wp_seq(x[:, self.obj_poses_indices])
        trajs = list_to_wp_seq(x[:, self.traj_indices])
        return np.concatenate([objs, trajs], axis=1)

    def apply_interaction(self, model, d, text,  label=False):
        data_new = []
        data_new.append({"input_traj": d["input_traj"], "output_traj": d["output_traj"], "text": text, "obj_names": d["obj_names"],
                        "obj_poses": d["obj_poses"]})

        X, _ = self.prepare_data(data_new, label=label)

        traj = list_to_wp_seq(X[:, self.traj_indices])
        traj_and_obj = self.prepare_x(X)
        emb = X[:, self.embedding_indices]
        source = [traj_and_obj, None, emb]
        pred_new = generate_2D(model, source).numpy()
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
