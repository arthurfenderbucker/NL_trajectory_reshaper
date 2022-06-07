import random
from typing import Generator
import numpy as np
import math

MAX_OBJS = 3
MAX_REP_R = 40.0
CHANGE_WEIGHT = 100.0
# BASE_COST = 10.0
# Causality + Objects + Types of changes + Change intensity

# Types of changes:
#   distance from obstacles: closer / further away from, keep a distance from x (3)
#   Speed changes: faster, slower, reduce/increase the speed (4)
#   Cartesian direction: stay on/go to the left/right/front/back  (8)
#   Sharpness / smoothness: make sharper/smoother/open/close turns/curves (8)




def repel(pts_raw,args, s=1.0,p=None,w=1.0):


    MAX_REP_R = 0.6
    MIN_REP_R = 0.05 
    base_w = 0.005

    pts = pts_raw[:,:3]
    diff = pts-p

    dist = np.expand_dims(np.linalg.norm(diff,axis=1),-1)
    dir = np.divide(diff, dist, out=np.zeros_like(diff), where=dist!=0)


    # f = np.divide(-s*(MAX_REP_R-dist), MAX_REP_R, out=np.zeros_like(dist), where=dist<MAX_REP_R)
    f = np.divide(-s*(MAX_REP_R/2), MAX_REP_R, out=np.zeros_like(dist), where=(dist>MIN_REP_R)&(dist<MAX_REP_R))

    F = np.zeros_like(pts_raw)
    F[:,:3] = dir*f*w*base_w

    return F


def cartesian_grad(pts_raw,args,dir=[1.0,0.0,0,0],center=[50,50], r = 60,w = 0.01):
   
    pts = pts_raw[:,:3]
    base_w = -0.2
    f = 1.0

    d = np.ones_like(pts)*dir
    F = np.zeros_like(pts_raw)
    F[:,:3] = d*f*w*base_w
    return F


#functions to change the map cost
# def repel(x,y,s=1.0,p=None,w=1.0):
#     # print("p = ",p)
#     cost = 1.0

#     d = math.hypot(p[0] - x, p[1] - y)
#     if d <= MAX_REP_R:
#         cost = np.maximum(1.0+s*(MAX_REP_R-d)/MAX_REP_R,0.01)# linear decay
    
#     return cost*w

def speed(pts_raw,args, s=1.0,p=None,w=1.0):

    MAX_REP_R = 0.6
    MIN_REP_R = 0.05 
    base_w = 0.005
    # print("v = ",value)

    if p is None:
        f = np.ones((pts_raw.shape[0],1))
    else:
        pts = pts_raw[:,:3] #3D
        # print(p)
        diff = pts-p

        dist = np.expand_dims(np.linalg.norm(diff,axis=1),-1)
        
        # dir = np.divide(diff, dist, out=np.zeros_like(diff), where=dist!=0)

        f = np.divide(-s*(MAX_REP_R-dist), MAX_REP_R, out=np.zeros_like(dist), where=dist<MAX_REP_R)

    F = np.zeros_like(pts_raw)
    F[:,3:] = f*w*base_w
    return F


# def cartesian_grad(x,y,dir=[1.0,0.0],center=[50,50], r = 60):
#     w = 10.0
#     norm = math.hypot(dir[0],dir[1])
#     cost = 1.0
#     d = ((x-center[0])*dir[0]+(y-center[1])*dir[1])/norm

#     # max_d  = (end[0]-start[0])*dir[0]+(end[1]-start[1])*dir[1]
#     max_d = 2*r
#     cost = 0
#     if(d>r):cost = 1.0
#     elif(d<-r):cost = 2.0
#     else: cost = np.clip(1.5+(-d)/abs(max_d),0.001,2.0)# linear decay
    
#     return cost

def get_map_cost(f_list):
    keys = {}
    for f in f_list:
        if isinstance(f, dict):
            for k, v in f.items():
                keys[k] = v
    def map_cost(pt,a):
        cost = 1.0
        for f in f_list:
            
            if isinstance(f, tuple): # function and value pair

                args = [f[1]]
                if len(f)>2:
                    args+=[keys[k] for k in f[2] if k in keys.keys()]
                    # print(f[2])
                    # print('keys: ',[k for k in f[2] if k in keys.keys()])
                    # print('args: ',args)
                
                cost *= f[0](pt,a,*args)
        return cost*CHANGE_WEIGHT
    return map_cost


class Label_generator:

    # Objects (max 3 per env):
    obj_names = ["table","chair","tree", "rock", "wall", "bycicle", "car", "computer", "cellphone", "TV"]
    obj_order = {"first","second","third","last"}
    objs = ["table","chair"]


    objs = {"table":{"value":{"obj_p":[1,1,1]}},
            "chair":{"value":{"obj_p":[2,2,2]}},
            "tree":{"value":{"obj_p":[3,3,3]}}}


    action_verbs = ["stay ", "pass ", "walk ", "drive "]


    # Change intensity:
        # Qualitative: a bit, a little,  much, very, - (5)
    change_intensity = {"a bit ":{"before":[],"after":[],"value":{"w":0.7}},
                        "a little ":{"before":[],"after":[],"value":{"w":0.7}},
                        "much ":{"before":[],"after":[],"value":{"w":1.5}},
                        "a lot ":{"before":[],"after":[],"value":{"w":1.5}},
                        "very ":{"before":[],"after":[],"value":{"w":1.5}},
                        "":{"before":[],"after":[],"value":{"w":1.0}}}

    dist_change = {"closer to the ":{"before":(action_verbs,change_intensity),"after":objs,"func":(repel,-1.0,["obj_p","w"])},
                "further away from the ":{"before":(action_verbs,change_intensity),"after":objs, "func":(repel,1.0,["obj_p","w"])},
                "keep a smaller distance from the ": {"before":[],"after":objs,"func":(repel,-1.0,["obj_p","w"])},
                "keep a bigger distance from the ": {"before":[],"after":objs,"func":(repel,1.0,["obj_p","w"])}}


    spatial_verbs = ["when passing ", "while passing "]   

    spatial_dep= {"close to the ":{"before":spatial_verbs,"after":objs},
                  "next to the ":{"before":spatial_verbs,"after":objs},
                  "near the ":{"before":spatial_verbs,"after":objs},
                  "nearby the ":{"before":spatial_verbs,"after":objs},
                  "in the surrounding of the ":{"before":spatial_verbs+[""],"after":objs},
                  "in the proximity of the ":{"before":spatial_verbs+[""],"after":objs},
                  "":{"before":[],"after":[]}}

    speed_verbs = ["walk ", "drive ", "go "]   
    speed_change = {"faster ":{"before":(speed_verbs, change_intensity),"after":spatial_dep,"func":(speed,1.0, ["obj_p","w"])},
                    "slower ":{"before":(speed_verbs, change_intensity),"after":spatial_dep,"func":(speed,-1.0, ["obj_p","w"])},
                    "reduce the speed ":{"before":[],"after":spatial_dep,"func":(speed,-1.0,["obj_p","w"])},
                    "increase the speed ":{"before":[],"after":spatial_dep,"func":(speed,1.0,["obj_p","w"])}}

    cartesian_verbs = ["stay on the ", "go to the ",""]
    cartesian_change = {"left":{"before":cartesian_verbs,"func":(cartesian_grad, [-1.0,0.0,0.0])},
                        "right":{"before":cartesian_verbs,"func":(cartesian_grad, [1.0,0.0,0.0])},
                        "front":{"before":cartesian_verbs,"func":(cartesian_grad, [0.0,1.0,0.0])},
                        "back":{"before":cartesian_verbs,"func":(cartesian_grad,[0.0,-1.0,0.0])},
                        "top":{"before":cartesian_verbs,"func":(cartesian_grad,[0.0,0.0,1.0])},
                        "down":{"before":["go","stay"],"func":(cartesian_grad,[0.0,0.0,-1.0])},
                        "upper part":{"before":cartesian_verbs,"func":(cartesian_grad,[0.0,0.0,1.0])},
                        "bottom part":{"before":cartesian_verbs,"func":(cartesian_grad,[0.0,0.0,-1.0])},
                        "bottom":{"before":cartesian_verbs,"func":(cartesian_grad,[0.0,0.0,-1.0])}}

    change_type = {"dist":(dist_change),"speed":(speed_change),"cartesian":(cartesian_change)}


    # Causality / Dependence:
        # Temporal: after x, before x, while passing through x (3) + none (1)
        # Spatial: close to x, next to x (2)
    temporal_dep = {"after":1,"before":-1}
    
    causality_type = {"temporal":[temporal_dep], "spatial":[spatial_dep],"":[None]}

    def __init__(self, objs, w=60, h=60):
        self.objs = objs
        self.h = h
        self.w = w

        # self.change_names = change_name

        for k in self.dist_change.keys():
            self.dist_change[k]["after"] = self.objs
        for k in self.spatial_dep.keys():
            if k != "":
                self.spatial_dep[k]["after"] = self.objs
        self.labels = []
        
    def generate_labels(self,change_names, shuffle=True):
        j = 0

        self.labels = []
        for c in change_names:
            for i,f in self.next_label(self.change_type[c], new=True):   
                # print("\n")
                # print(i)
                # print("=====> ",i, " --- ",f)
                map_cost = get_map_cost(f)
                
                self.labels.append([i,map_cost])
        random.shuffle(self.labels)
        

    def sample_labels(self, n):
        return random.sample(self.labels, k=n)

    def next_label(self, p, func_list = [], new = False): #all the combinations of labels
        if new: func_list = []
        # global i 
        # print(i)
        # i += 1
        # print(p)
        text = ""
        
        if isinstance(p, tuple) and len(p)>0:
            for p1,f1 in self.next_label(p[0],func_list=func_list):
                for p2,f2 in self.next_label(p[1], func_list=func_list):
                    f = []
                    if f1: f += f1
                    if f2: f += f2
                    yield (p1 + p2, f)
            
        elif isinstance(p, list) and len(p)>0:
            # yield from p
            for i in p:
                yield (i,None)
        elif isinstance(p, dict):

            for c_text,c_value in p.items():
            # c_text,c_value  = random.choice(list(p.items()))
                b_gen = [("",None)]
                if "before" in c_value.keys() and len(c_value["before"])>0:
                    b_gen = self.next_label(c_value["before"],func_list=func_list)

                for b,fb in b_gen:
                    text = b+c_text
                    a_gen = [("",None)]
                    if "after" in c_value.keys() and len(c_value["after"])>0:
                        a_gen = self.next_label(c_value["after"], func_list=func_list)
                    for a,fa in a_gen:

                        fl = []
                        if fb: fl+=fb
                        if fa: fl+=fa

                        if "func" in c_value.keys():
                            fl.append(c_value["func"])
                        elif "value" in c_value.keys():
                            fl.append(c_value["value"])
                        
                        yield (text + a, fl)

f_list = []
def recursive_label(p):
    global f_list
    text = ""
    if isinstance(p, tuple) and len(p)>0:
        for pi in p:
            text += recursive_label(pi)
    if isinstance(p, list) and len(p)>0:
        text = random.choice(p)
    elif isinstance(p, dict):
        c_text,c_value  = random.choice(list(p.items()))

        if "before" in c_value.keys() and len(c_value["before"])>0:
            text += recursive_label(c_value["before"])
        text += c_text
        if "after" in c_value.keys() and len(c_value["after"])>0:
            text += recursive_label(c_value["after"])
        if "func" in c_value.keys() and "value" in c_value.keys():
            f_list.append((c_value["func"],c_value["value"]))
        elif "value" in c_value.keys():
            f_list.append(c_value["value"])
    return text



def main():
    global f_list
    # obj_names = random.sample(obj_names, random.randint(0,MAX_OBJS))

    lg = Label_generator({"table":{"value":{"obj_p":[1,1]}},
                        "chair":{"value":{"obj_p":[2,2]}},
                        "tree":{"value":{"obj_p":[4,4]}}})
    lg = Label_generator({"object X":{"value":{"obj_p":[1,1]}}})
    lg.generate_labels()
    # j = 0
    # for i,f in lg.next_label(change_type["dist"], new=True):
    #     # print("\n")
    #     print("=====> ",i, " --- ",f)
    #     map_cost = perform(f)
    #     print(map_cost(6,0))
    #     # print("      ",len(f))
    #     j+=1
    # print(j)
    
    # cartesian_grad(10,10,[1.0,0.0])
    # f_list = [(cartesian_grad,[[1.0,0.0],[0,5],[10,10]])]
    # f_list = [(repel,[1.0,[10,10]])]

    # print(perform(10,5))

    # print(len(f_list))
    # for i in range(10):
    #     print(recursive_label(change_type["dist"]))

if __name__ == '__main__':
    main()
