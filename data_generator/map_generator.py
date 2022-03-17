import numpy as np
import matplotlib.pyplot as plt
import math
import random
# import sys
# sys.path.append("../PythonRobotics")
# from a_star import AStarPlanner 
# from PathPlanning.DStarLite import d_star_lite as m

MAX_OBJS = 3
MIN_OBJ_R = 5
MAX_OBJ_R = 10

show_map = True

class Map_generator:
    def __init__(self, s=[10.0,10.0], g=[50.0,50.0],d=[60,60]):

        #map settings
        self.sx, self.sy = s 
        self.gx,self.gy = g 
        self.h,self.w = d
        self.o_center_x, self.o_center_y = [],[]
        self.ox, self.oy = [], []


    def plot_map(self, show = False):
        plt.plot(self.ox, self.oy, ".k")
        plt.plot(self.sx, self.sy, "og")
        plt.plot(self.gx, self.gy, "xb")

        plt.grid(True)
        plt.axis("equal")

        if show:
            plt.show()


    def generate_objects(self, n_objs  = 3):
        # Walls

        for i in range(0, self.w+1):
            self.ox.append(i)
            self.oy.append(0)
        for i in range(0, self.h+1):
            self.ox.append(self.w)
            self.oy.append(i)
        for i in range(0, self.w+1):
            self.ox.append(i)
            self.oy.append(self.h)
        for i in range(0, self.h+1):
            self.ox.append(0)
            self.oy.append(i)

        # for i in range(int(self.h/3.0),self.h+1):

        #     self.ox.append(int(self.w*2.0/3.0)-1)
        #     self.oy.append(i)

        # for d in range(int(self.h/20.0)):
        #     for i in range(0, int(self.h*2/3.0)+1):
        #         self.ox.append(d+int(self.w/3.0))
        #         self.oy.append(i)


        for n_i in range(n_objs):
            r = random.randint(MIN_OBJ_R, MAX_OBJ_R)
            
            margin = int(self.w/6)
            x = random.randint(r+margin, self.w-r-margin)
            y = random.randint(r+margin, self.h-r-margin)
            if math.dist([self.sx,self.sy],[x,y]) <= r or math.dist([self.gx,self.gy],[x,y]) <= r:
                # print(n_i)
                # n_i-=1
                continue

            self.o_center_x.append(x)
            self.o_center_y.append(y)
            for i in range(x-r,x+r+1):
                for j in range(y-r,y+r+1):
                    if math.dist([i,j],[x,y]) > r:continue
                    self.ox.append(i)
                    self.oy.append(j)
    

    def get_next_map(self):
        self.reset_vars()
        out = {"o_center_x": self.o_center_x, "o_center_y": self.o_center_y, "ox": self.ox, "oy": self.oy}
        return out

    def reset_vars(self):
        self.o_center_x, self.o_center_y = [],[]
        self.ox, self.oy = [], []
        self.generate_objects()


def main():
    print(__file__ + " start!!")


    mg = Map_generator()
    mg.get_next_map()

    # def map_cost(x, y):
    #     cost = 0
    #     w = 5.0
    #     for rep_p in rep_points:
    #         d = math.hypot(rep_p[0] - x, rep_p[1] - y)
    #         if d <= rep_p[2]:
    #             cost += w*(rep_p[2]-d) # linear decay
    #     return cost

    # a_star = AStarPlanner(ox, oy, grid_size, robot_radius, map_cost = map_cost)
    # rx, ry = a_star.planning(sx, sy, gx, gy)

    # if show_animation:  # pragma: no cover
    #     plt.plot(rx, ry, "-r")
    #     plt.pause(0.001)
    #     plt.show()


    # set obstacle positions
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)




if __name__ == '__main__':
    main()
