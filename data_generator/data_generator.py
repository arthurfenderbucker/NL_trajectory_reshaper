from map_generator import Map_generator
from a_star import AStarPlanner
from labels_generator import Label_generator


import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil

CHANGE_WEIGHT = 100.0
MAX_COST = 200.0
data_dir = 'train_data/'
def ones(x, y):
    cost = CHANGE_WEIGHT
    return cost


def main():
    print(__file__ + " start!!")

    show_animation = False

    grid_size = 3.0
    robot_radius = 1.0

    sx, sy= 10.0,10.0
    gx, gy= 90.0,90.0
    w, h = 100,100

    mg = Map_generator(s=[sx,sy], g=[gx,gy],d=[w,h])
    
   
    def plot_map_cost(map_cost):
        max_d = CHANGE_WEIGHT
    
        plt.imshow(map_cost,origin='lower')
        plt.colorbar()
   
    n_maps = 100
    for map_id in range(0,n_maps):
        
        print(map_id)
        m = mg.get_next_map()
    
        all_obj_names = ["table","chair","tree", "rock", "wall", "bycicle", "car", "computer", "cellphone", "TV"]
        obj_names = random.sample(all_obj_names, k = len(m["o_center_x"]))
        objs  = {}
        for x,y,name in zip(m["o_center_x"],m["o_center_y"],obj_names):
            objs[name] = {"value":{"obj_p":[x,y]}}
        # print(objs)

        total = 0

        lg = Label_generator(objs,w=w,h=h)
        
        change_type = ['dist','cartesian']
        if map_id > n_maps*0.7:
            change_type = ['dist']

        lg.generate_labels(change_type)
        # lg.generate_labels('speed')
        
        a_star = AStarPlanner(m["ox"], m["oy"], grid_size, robot_radius, map_cost=ones)
        rx_original, ry_original = a_star.planning(sx, sy, gx, gy)
        
        path = os.path.join(data_dir, str(map_id))
        if os.path.exists(path): shutil.rmtree(path)
        os.mkdir(path)
        
        for text, map_cost_f in lg.sample_labels(5):
            
            map_cost = np.array([[map_cost_f(x,y) for x in range(w+1)] for y in range(h+1)])
            # map_cost[m["oy"],m["ox"]] = MAX_COST
            
            if show_animation:  # pragma: no cover
                plt.figure()
                # print("\n",text)
                plt.title(text)
                plot_map_cost(map_cost)
            
            #save map
            np.savetxt(os.path.join(path,text+".csv"), map_cost, delimiter=",")

            with open(os.path.join(path,"META.json"), 'w') as fp:

                map_objs_only = np.ones_like(map_cost)
                map_objs_only[m["oy"],m["ox"]] = 0
                map_objs_only_list = map_objs_only.tolist()

                data = {"map":map_objs_only_list,"width":w, "height":h, "p_start":[sx, sy], "p_goal":[gx, gy] ,"o_center_x":m["o_center_x"],"o_center_y":m["o_center_y"],"obj_names":obj_names, "rx_original":rx_original, "ry_original":ry_original}
                json.dump(data, fp)


            # mg.plot_map()

            if show_animation:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                for x,y,name in zip(m["o_center_x"],m["o_center_y"],obj_names):
                    plt.text(x, y, name, bbox=props)

                # plt.show(block=False)

                # plt.figure()
                # a_star = AStarPlanner(m["ox"], m["oy"], grid_size, robot_radius, map_cost = map_cost)
                # mg.plot_map()
                # rx, ry = a_star.planning(sx, sy, gx, gy)


                # plt.plot(rx_original, ry_original, "-b")
                # plt.plot(rx, ry, "-g")
                plt.pause(0.001)
                plt.show(block=False)
            




if __name__ == '__main__':
    main()
