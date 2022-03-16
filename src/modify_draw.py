import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import numpy as np
import json

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import rospy
from geometry_msgs.msg import PoseStamped
import os
import math

import argparse

from traj_utils import *
from simple_TF_continuos import *
from motion_refiner import Motion_refiner
from functions import *


parser = argparse.ArgumentParser(description='collect traj draw.')
parser.add_argument('--name', type=str, default="user")
parser.add_argument('--trial', type=int, default=1)
parser.add_argument('--img_file', type=str, default="/home/gari/ros_ws/src/motion_refiner/top_view.png")

args = parser.parse_args()

original_traj = np.load(
    "/home/gari/ros_ws/src/motion_refiner/original_traj.npy")

base_path = "/home/gari/ros_ws/src/motion_refiner/user_trajs/"


class Drawing_interface():
    def __init__(self, user_name="user", img_file=args.img_file):

        self.img_file = img_file
        self.crop = np.array([[25, 25], [746, 406]])

        self.img = cv2.imread(self.img_file)
        self.base_img = cv2.imread(self.img_file)

        self.ix = -1
        self.iy = -1
        self.drawing = False
        self.placing_objs = True
        self.record_interactios = True
        self.interaction_type = ""
        self.user_name = args.name

        self.trial = int(args.trial)
        self.base_path = base_path

        self.obj_i = 0
        self.obj_poses = [[477, 64], [291, 81], [271, 313]]
        self.obj_names = ["glasses", "cellphone", "wine"]

        self.points = []
        self.new_traj = []
        self.simple_traj = []

        self.last_trajs = []

        cv2.namedWindow(winname="Motion refiner")
        cv2.setMouseCallback("Motion refiner",
                             self.event_cb)

        self.text = ""
        # (3, 52, 89)  # (184, 188, 191)  # (14, 130, 207)
        self.obj_color = (204, 3, 255)
        self.traj_color = (0, 0, 255)
        self.new_traj_color = (0, 255, 0)
        self.new_traj_color_old = (0, 100, 0)

        self.draw_new_traj = True
        self.simple_obj_poses = []

    def draw_objs(self):
        for i, p in enumerate(self.obj_poses):
            cv2.circle(self.img, (int(p[0]), int(p[1])), 5, self.obj_color, -1)
            self.img = cv2.putText(self.img, self.obj_names[i], (int(p[0])-20, int(p[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5,  self.obj_color, 1, cv2.LINE_AA)

    def draw_traj(self, traj, color):
        for i in range(len(traj)):
            p = traj[i]
            cv2.circle(self.img, (int(p[0]), int(
                p[1])), 2, color, -1)
            if i > 0:
                cv2.line(self.img, (int(traj[i-1][0]), int(traj[i-1][1])),
                         (int(p[0]), int(p[1])), color, 3)

    def draw_interaction(self):
        self.img = cv2.putText(self.img, self.text, (50, 25), cv2.FONT_HERSHEY_SIMPLEX,
                               0.75,  (255, 255, 255), 2, cv2.LINE_AA)

    def redraw(self):
        self.img = self.base_img.copy()
        self.draw_objs()
        self.draw_traj(self.points, self.traj_color)
        for i in range(len(self.last_trajs)):
            self.draw_traj(self.new_traj, self.new_traj_color_old)
        self.draw_traj(self.new_traj, self.new_traj_color)

        self.draw_interaction()

    def event_cb(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.placing_objs:
                self.obj_poses[self.obj_i] = [x, y]
                print("obj poses: ", self.obj_poses)
                self.redraw()
                self.drawing = False

            else:
                self.drawing = True
                ix = x
                iy = y
                print((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                # cv2.rectangle(img, pt1=(ix, iy),
                #               pt2=(x, y),
                #               color=(0, 255, 255),
                #               thickness=-1)

                if not self.draw_new_traj:
                    cv2.circle(self.img, (x, y), 2, self.traj_color, -1)
                    if len(self.points) > 0:
                        cv2.line(self.img, (self.points[-1][0], self.points[-1][1]),
                                 (x, y), self.traj_color, 3)
                    self.points.append([x, y])
                else:

                    cv2.circle(self.img, (x, y), 2, self.new_traj_color, -1)
                    if len(self.points) > 0:
                        cv2.line(self.img, (self.points[-1][0], self.points[-1][1]),
                                 (x, y), self.new_traj_color, 3)
                    self.new_traj.append([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # cv2.rectangle(img, pt1=(ix, iy),
            #               pt2=(x, y),
            #               color=(0, 255, 255),
            #               thickness=-1)

    def save_interaction(self):
        file_path = os.path.join(
            self.base_path, self.interaction_type, self.user_name, str(self.trial))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        print(file_path)

        np.save(file_path+"_base_traj.npy", np.array(self.points))
        np.save(file_path+"_new_traj.npy", np.array(self.new_traj))
        with open(file_path+"_text-"+self.text+".txt", "w") as text_file:
            text_file.write(self.text)
        cv2.imwrite(file_path+"_img.png", self.img)
        print("user: "+self.user_name+"\ttrial: "+str(self.trial))
        self.trial += 1
        print("interaction saved")

    def show(self):
        # cv2.imshow("Motion refiner", self.img[self.crop[0, 1]:self.crop[1, 1], self.crop[0, 0]:self.crop[1, 0]])
        cv2.imshow("Motion refiner", self.img)

        # thresh = cv2.threshold(cv2.cvtColor(self.m, cv2.COLOR_BGR2GRAY),
        #                        cv2.getTrackbarPos('thresh', 'ctrl'),
        #                        255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('thresh', thresh)

    def reset_points(self):
        self.last_trajs.append(self.points)
        self.points = []
        self.new_traj = []

        self.redraw()
        print("reset")

    def update_traj(self):
        nt = self.new_traj
        self.reset_points()
        self.points = nt
        print("traj updated")
        self.redraw()

    def get_env(self):
        return np.array(self.points), np.array(self.obj_poses), self.text, self.obj_names

    def set_text(self, text):
        self.text = text
        di.redraw()
        print("text updated")

    def set_image(self, cv_image):
        self.base_img = cv_image
        di.redraw()

    def set_user(self, name):
        self.user_name = name

    def set_trial(self, trial):
        self.trial = trial


def publish_simple_traj(traj, pub):
    msg = Path()
    msg.header.frame_id = "/panda_link0"
    msg.header.stamp = rospy.Time.now()
    for wp in traj:
        ps = PoseStamped()

        ps.pose.position.x = wp[0]
        ps.pose.position.y = -wp[1]
        ps.pose.position.z = 0
        msg.poses.append(ps)

    pub.publish(msg)


def load_traj(file):
    with open(file) as f:
        d = json.load(f)
    return d


def apply_interaction(mr, traj, obj_poses_, text, obj_names, n_obj=3):

    mod = 1
    p1 = np.array([0.1, 0.1])-0.5
    p2 = np.array([0.9, 0.9])-0.5

    traj_np = traj
    if isinstance(traj, list):
        traj_np = np.array(traj)

    obj_poses = obj_poses_
    if isinstance(obj_poses_, list):
        obj_poses = np.array(obj_poses_)

    # if obj_poses.shape[0]!=n_obj:

    traj_raw = traj_np[::mod, :2]
    t0 = traj_raw[0, :]
    tf = traj_raw[-1, :]

    # vprint("traj_raw:", limits(traj_raw))

    traj_raw_n, obj_poses_new = interpolate_2points(
        traj_raw, p1, p2, objs=obj_poses.T)

    # vprint(limits(traj_raw_n))
    # print(obj_poses.T)

    d = np2data(traj_raw_n, obj_names,
                obj_poses_new.T, text, output_traj=None)
    # vprint(d)
    X, _ = mr.prepare_data(d, label=False)
    # vprint(limits(X))

    pred, traj_in = mr.apply_interaction(
        model, d[0], text,  label=False)
    # vprint(pred.shape)
    # vprint(limits(pred))

    print(pred)
    print(pred.shape)
    print(t0, tf)

    # cnt = np.ones([traj_in.shape[0]])*0.10
    # pred_cnt = mr.follow_hard_constraints(
    #     traj_in, pred[0, :, :], cnt)
    new_traj_simple = interpolate_2points(pred[0, :, :], t0, tf)

    # x, y = interpolate_traj(new_traj_simple, n=1000)
    new_traj_wp = fit_wps_to_traj(new_traj_simple, traj_raw)

    constraints = np.ones([traj_raw.shape[0]])*0.15

    new_traj_wp_cnt = mr.follow_hard_constraints(
        traj_raw, new_traj_wp, constraints)

    new_traj_wp_scaled = mr.addapt_to_hard_constraints(
        traj_raw, new_traj_wp, constraints)

    # pred_t = np.transpose(pred[:, :, :2], [0, 2, 1])
    # pred_d = pred_t.reshape([pred_t.shape[0], pred_t.shape[2]*2])

    # show_data(d, pred=pred_d, abs_pred=True, block=False,
    #           show_label=False, new_fig=True,  show_interpolated=True, show_original=False)
    # plt.pause(.001)

    return new_traj_wp, pred[0, :, :], obj_poses_new.T


def modify_traj(mr, di):
    traj, obj_poses, text, obj_names = di.get_env()
    new_traj, simple_traj, simple_obj_poses = apply_interaction(
        mr, traj, obj_poses.T, text, obj_names)

    di.new_traj = new_traj.tolist()
    di.simple_traj = simple_traj.tolist()
    di.simple_obj_poses = simple_obj_poses.tolist()


rospy.init_node('draw_interface', anonymous=True)
traj_pub = rospy.Publisher("/simple_traj", Path)
objs_pub = rospy.Publisher("/obj_poses", Path)


# model = load_model(
#     "/home/gari/data/models/exp_norm_layer_tanh_warmup/TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:3&concat_emb:True&features_n:777&optimizer:adam&norm_layer:True&activation:tanh.h5")
model_path = "/home/gari/data/models/exp_optmizer_depth/"
model_name = "refined_refined_TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:3&concat_emb:True&features_n:777&optimizer:RMSprop&norm_layer:True&activation:tanh.h5"
model_file = model_path + model_name

load_models = True
if load_models:
    model = load_model(model_file)

    compile(model)
mr = Motion_refiner(load_models=load_models)

di = Drawing_interface()
di.set_text("stay further away from the glasses")


bridge = CvBridge()
live_image = True
image_topic = "/camera_top/color/image_raw"

while not rospy.is_shutdown():

    if live_image:
        data = rospy.wait_for_message(image_topic, Image)
        try:
            cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        (rows, cols, channels) = cv_image.shape
        if not (cols > 60 and rows > 60):  # returns if data have unvalid shape
            continue
        di.set_image(cv_image)
    di.show()

    h_, w_ = di.img.shape[:2]
    w, h = 100.0, 100.0
    fx, fy = w/w_, h/h_

    k = cv2.waitKey(10)
    if k == 27:
        break
    elif k == ord("d"):
        di.reset_points()

    elif k == ord("t"):
        di.placing_objs = False

    elif k == ord("w"):
        di.interaction_type = "chomp"
        file_path = os.path.join(
            "/home/gari/data/data/user_exp/modified/", di.user_name+".json")
        print("\nloading: ", file_path)
        di.placing_objs = False
        di.draw_new_traj = True
        try:
            traj = 100*np.array(load_traj(file_path)["output_traj"])/[fx, fy]
            print(traj)
            di.new_traj = traj.tolist()
            di.redraw()
        except:
            print("file not found!!")

    elif k == ord("i"):
        print("\n\n Please input an interaction text and click ENTER:")
        text = input()
        di.set_text(text)
        modify_traj(mr, di)
        di.redraw()

    elif k == ord("l"):

        traj = np.concatenate(
            [np.array(di.obj_poses), np.array(di.new_traj)], axis=0)
        publish_simple_traj(traj, traj_pub)
        print("traj and objs published")

    elif k == ord("p"):
        traj = np.concatenate(
            [np.array(di.obj_poses), np.array(di.points)], axis=0)
        publish_simple_traj(traj, traj_pub)
    elif k == ord("u"):
        di.update_traj()
    elif k == ord("o"):
        di.points = original_traj[1:].tolist()
        di.redraw()

    elif k == ord("n"):
        di.placing_objs = False
        di.draw_new_traj = True
        di.points = original_traj[1:].tolist()
        di.redraw()
    elif k == ord("m"):

        modify_traj(mr, di)

    elif k == ord("a"):
        print("\nwrite the user name and press ENTER")
        name = input()
        di.set_user(name)

        print("\nwrite the trial and press ENTER")
        trial = int(input())
        di.set_trial(trial)
    elif k == ord("q"):
        di.save_interaction()

    elif k == ord("y"):
        di.record_interactios = True
        di.interaction_type = "NL"
        print("interaction type: ", di.interaction_type)
        di.set_trial(1)

    elif k == ord("x"):
        di.record_interactios = True
        di.interaction_type = "drawing"
        print("interaction type: ", di.interaction_type)
        di.set_trial(1)

    elif k == ord("s"):
        print(di.img.shape[:2])
        # h, w = h_, w_
        # fx, fy = 1.0, 1.0

        traj = np.array(di.points)*[fx, fy]
        objs = np.array(di.obj_poses)*[fx, fy]
        print(fx, fy)
        sx, sy = traj[0, :]*[fx, fy]
        gx, gy = traj[-1, :]*[fx, fy]

        r = 5
        ox = []
        oy = []
        for o in objs:
            print("obj ", o)
            x, y = int(o[0]), int(o[1])
            for i in range(x-r, x+r+1):
                for j in range(y-r, y+r+1):
                    if math.dist([i, j], [x, y]) > r or i >= w or j >= h or i < 0 or j < 0:
                        continue
                    ox.append(i)
                    oy.append(j)
        print(w/h)
        print(max(ox), max(oy))

        data_path = "/home/gari/data/data/user_exp/0/"
        f = os.path.join(data_path, "META.json")
        with open(f, 'w') as fp:

            map_objs_only = np.ones([int(h), int(w)])
            map_objs_only[oy, ox] = 0
            map_objs_only_list = map_objs_only.tolist()

            data = {"map": map_objs_only_list, "width": int(w), "height": int(h), "p_start": [int(sx), int(sy)], "p_goal": [int(gx), int(gy)],
                    "o_center_x": objs[:, 0].tolist(), "o_center_y": objs[:, 1].tolist(), "obj_names": di.obj_names,
                    "rx_original": traj[:, 0].tolist(), "ry_original": traj[:, 1].tolist(), "robot_base": [int(400*fx), int(350*fy)]}
            # print(data)
            json.dump(data, fp)
        print("file saved! ", f)

    elif k >= ord("1") and k <= ord("3"):
        di.obj_i = k-ord("1")
        di.placing_objs = True
    # if k != -1:
    #     print(k)
# np.save(user_file+".npy", np.array(points))
# cv2.imwrite(user_file+".png", img)

cv2.destroyAllWindows()
