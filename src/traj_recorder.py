#!/usr/bin/env python
from keras.layers import BatchNormalization, GlobalAveragePooling1D, Embedding, Flatten, Layer, Dense, Dropout, MultiHeadAttention, Attention, Conv1D, Input, Lambda, Concatenate, LayerNormalization
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from keras import Model
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
import rospy
from std_msgs.msg import String, Empty, Bool
from my_interfaces.msg import MoveEEToPoseActionGoal, DualQuaternion, MoveEEToPoseAction, MoveEEToPoseGoal
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
import actionlib
import actionlib_tutorials.msg
import math
from dqrobotics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
import numpy as np
import json


from motion_refiner.traj_utils import *
from motion_refiner.src.simple_TF_continuos import *
from motion_refiner.src.motion_refiner import Motion_refiner
from motion_refiner.src.functions import *


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


verbose = True
vprint = print if verbose else lambda *a, **k: None


def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


vprint(get_available_devices())


def obj_piramid(file, pos, ori, n=3, id0=0, spacing=[0.05125, 0.05125, 0.125]):
    marker_list = []
    for i in range(n):
        for j in range(n-i):
            for k in range(n-i-j):
                x = pos[0] - (j * spacing[1]+i*spacing[0]/2)
                y = pos[1] + (k * spacing[0]+j*spacing[1]/2 + i*spacing[0]/2)
                z = (i+0.5) * spacing[2]
                marker_list.append(get_marker(
                    file, [x, y, z], ori, id0+i*n*n+j*n+k, scale=[1.25, 1.25, 1.0]))
    return marker_list


def get_marker(file, pos, ori, id=0, scale=[1, 1, 1]):
    marker = Marker()

    marker.header.frame_id = "/panda_link0"
    marker.header.stamp = rospy.Time.now()

    # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
    marker.type = marker.MESH_RESOURCE
    marker.id = id

    # Set the scale of the marker
    marker.scale.x = 0.001*scale[0]
    marker.scale.y = 0.001*scale[1]
    marker.scale.z = 0.001*scale[2]

    # Set the color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.pose.position.x = pos[0]
    marker.pose.position.y = pos[1]
    marker.pose.position.z = 0 if len(pos) == 2 else pos[2]

    marker.pose.orientation.x = ori[0]
    marker.pose.orientation.y = ori[1]
    marker.pose.orientation.z = ori[2]
    marker.pose.orientation.w = ori[3]

    marker.action = marker.ADD

    marker.mesh_use_embedded_materials = True
    marker.mesh_resource = file
    return marker


class Motion_refiner_wrapper():

    def __init__(self, load_model=True):

        rospy.init_node('traj_recorder', anonymous=True)

        self.traj_sub = rospy.Subscriber(
            "goToPose/goal", MoveEEToPoseActionGoal, self.callback)

        # self.traj_pub=rospy.Publisher("goToPose/goal", MoveEEToPoseActionGoal,queue_size=200)

        rospy.Subscriber('apply_interaction', String, self.apply_interaction)
        rospy.Subscriber('save_traj', String, self.save_traj)
        rospy.Subscriber('record_traj', Bool, self.record_traj)
        rospy.Subscriber('load_traj', String, self.load_traj)
        rospy.Subscriber('execute_new_traj', Empty, self.execute_new_traj)
        rospy.Subscriber('reset_traj', Empty, self.reset_traj)

        rospy.Subscriber('simple_traj', Path, self.simple_traj_cb)
        rospy.Subscriber('obj_poses', Path, self.obj_poses_traj_cb)

        rospy.Subscriber('go_to_init', String, self.go_to_init)

        self.obj_names = ["glasses", "cellphone", "wine"]
        # self.obj_poses = np.array([[0.5673348043972765, 0.4143015029356061, 0.3704927224189868],
        #                            [-0.19159203849854456, 0.5597855247159429, -0.6584785035263063]])

        # self.obj_poses = np.array([[0.5673348043972765, 0.1143015029356061, 0.3704927224189868],
        #                            [-0.19159203849854456, 0.1597855247159429, -0.6584785035263063]])
        self.obj_poses = np.array([[0.8673348043972765, 1.0, 1.0],
                                   [-0.19159203849854456, 0.4, -0.25]])
        self.obj_ori = np.array([[0.0, 0.0, 0.707],
                                 [0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0],
                                 [1.0, 1.0, 0.707]])
        # self.stl_files = ["phone.stl", "cup.stl", "bottle.stl"]
        self.stl_files = ["glass.stl", "phone.stl", "bottle.stl"]

        self.constraints = None
        self.marker_pub = rospy.Publisher("/objs_markers", MarkerArray)
        self.traj_pub = rospy.Publisher("/traj_original", Path)
        self.new_traj_pub = rospy.Publisher("/traj_new", Path)
        # self.traj_constraints_pub = rospy.Publisher("/traj_constraints", Path)

        self.new_traj_no_constraints_pub = rospy.Publisher(
            "/traj_new_no_constraints", Path)

        self.fig = plt.figure(figsize=[5, 5])

        self.record = False

        # self.traj_bottle_up = self.get_traj_from_file(file_name='bottle_up.json')
        # self.traj_pou = self.get_traj_from_file(file_name='bottle_up.json')
        self.traj_trim_end = 140
        self.traj_trim_start = 15

        self.reset_traj()

        # self.model = load_model(
        #     "/home/gari/data/models/test_refine/refined_TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:4&dropout_rate:0.1&wp_d:2.h5")
        # "TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2.h5")
        # self.model = load_model(
        #     "/home/gari/data/models/exp_depth_refining/refined_TF&num_layers_enc:1&num_layers_dec:6&d_model:256&dff:1024&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:5&concat_emb:True&features_n:777.h5")

        # self.model = load_model(
        #     "/home/gari/data/models/exp_norm_layer/TF&num_layers_enc:1&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:3&concat_emb:True&features_n:777&optimizer:adam&norm_layer:True.h5")

        if load_model:
            self.model = load_model(
                "/home/gari/data/models/exp_norm_layer_tanh_warmup/TF&num_layers_enc:2&num_layers_dec:4&d_model:256&dff:512&num_heads:8&dropout_rate:0.1&wp_d:2&bs:64&dense_n:512&num_dense:3&concat_emb:True&features_n:777&optimizer:adam&norm_layer:True&activation:tanh.h5")
            compile(self.model)
            self.mr = Motion_refiner()
        else:
            self.mr = Motion_refiner(load_models=False)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.pub_markers)

    def pub_markers(self, timer):

        markerArray = MarkerArray()

        for i in range(len(self.obj_names)):
            if self.obj_names[i] == "glasses":
                obj_list = obj_piramid("package://motion_refiner/meshes/" +
                                       self.stl_files[i], self.obj_poses[:, i], self.obj_ori[:, i], id0=i+100)
                for o in obj_list:
                    markerArray.markers.append(o)
            else:
                markerArray.markers.append(get_marker(
                    "package://motion_refiner/meshes/" + self.stl_files[i], self.obj_poses[:, i], self.obj_ori[:, i], id=i))

        self.marker_pub.publish(markerArray)
        self.visualize_new_traj()
        self.visualize_traj()
        self.visualize_new_traj_no_constraints()
        # if not self.constraints is None:
        #     self.visualize_traj_constraints()
        # rospy.loginfo("markers published")

    def visualize_traj(self):
        msg = Path()
        msg.header.frame_id = "/panda_link0"
        msg.header.stamp = rospy.Time.now()
        for i, wp in enumerate(self.traj):
            ps = PoseStamped()
            ps.pose = dqmsg2pose(wp.goal.target_pose)

            msg.poses.append(ps)

        self.traj_pub.publish(msg)

    # def visualize_traj_constraints(self):
    #     msg = Path()
    #     msg.header.frame_id = "/panda_link0"
    #     msg.header.stamp = rospy.Time.now()
    #     for i, wp in enumerate(self.traj):
    #         ps = PoseStamped()
    #         ps.pose = dqmsg2pose(wp.goal.target_pose)
    #         msg.poses.append(ps)

    #     self.traj_pub.publish(msg)

    def visualize_new_traj(self):
        msg = Path()
        msg.header.frame_id = "/panda_link0"
        msg.header.stamp = rospy.Time.now()
        for i, wp in enumerate(self.new_traj):
            ps = PoseStamped()
            ps.pose = dqmsg2pose(wp.target_pose)
            msg.poses.append(ps)

        self.new_traj_pub.publish(msg)

    def visualize_new_traj_no_constraints(self):
        msg = Path()
        msg.header.frame_id = "/panda_link0"
        msg.header.stamp = rospy.Time.now()
        for i, wp in enumerate(self.new_traj_no_constraints):
            ps = PoseStamped()
            ps.pose = dqmsg2pose(wp.target_pose)
            msg.poses.append(ps)

        self.new_traj_no_constraints_pub.publish(msg)
    # ============ CALBACKS =============

    def obj_poses_traj_cb(self, data):
        objs = path2npcoords(data)
        print("OBJS!!")
        print(objs)
        self.obj_poses = objs.T

    def simple_traj_cb(self, data, num_objs=3):

        if len(self.traj) == 0:
            rospy.loginfo("no traj loaded, traj len = "+str(len(self.traj)))
            return
        rospy.loginfo("simple traj received")

        wp_received = path2npcoords(data)
        obj_poses = wp_received[:num_objs, :]

        received_traj_np = wp_received[num_objs:, :]
        print(received_traj_np.shape)

        traj_np = wplist2np(self.traj)[
            self.traj_trim_start:self.traj_trim_end]
        mod = 4
        traj_raw = traj_np[::mod, :2]
        t0 = traj_raw[0, :]
        tf = traj_raw[-1, :]

        new_traj_simple, obj_poses_new = interpolate_2points(
            received_traj_np, t0, tf, objs=obj_poses)
        print(obj_poses)
        print(obj_poses_new.T)
        self.obj_poses = obj_poses_new.T
        new_traj_wp = fit_wps_to_traj(new_traj_simple, traj_raw)

        # x, y = interpolate_traj(new_traj_simple, n=1000)
        new_traj_wp = fit_wps_to_traj(new_traj_simple, traj_raw)

        self.constraints = np.ones([traj_raw.shape[0]])*0.15

        new_traj_wp_cnt = self.mr.follow_hard_constraints(
            traj_raw, new_traj_wp, self.constraints)

        new_traj_wp_scaled = self.mr.addapt_to_hard_constraints(
            traj_raw, new_traj_wp, self.constraints)

        self.new_traj_no_constraints = self.update_traj(
            self.traj, new_traj_wp, mod=mod)
        self.new_traj = self.update_traj(
            self.traj, new_traj_wp_scaled, mod=mod)

        # self.publish_traj(self.new_traj)

    def apply_interaction(self, data):
        # rospy.loginfo(data.data + " "+str(len(self.traj)))
        print(data)
        text = ""
        if isinstance(data, str):
            text = data
        elif hasattr(data, 'data'):
            text = data.data

        vprint(text)
        new_traj_wp = None
        if len(text) > 0:
            # if len(self.new_traj) > 0:
            #     self.traj = self.new_traj
            mod = 4
            p1 = np.array([0.1, 0.1])-0.5
            p2 = np.array([0.9, 0.9])-0.5
            traj_np = wplist2np(self.traj)[
                self.traj_trim_start:self.traj_trim_end]
            print("traj_np:", limits(traj_raw))

            traj_raw = traj_np[::mod, :2]
            t0 = traj_raw[0, :]
            tf = traj_raw[-1, :]

            print("traj_raw:", limits(traj_raw))

            traj_raw_n, obj_poses_new = interpolate_2points(
                traj_raw, p1, p2, objs=self.obj_poses.T)

            vprint(limits(traj_raw_n))
            # print(self.obj_poses.T)

            d = np2data(traj_raw_n, self.obj_names,
                        obj_poses_new.T, text, output_traj=None)
            # vprint(d)
            X, _ = self.mr.prepare_data(d, label=False)
            vprint(limits(X))

            pred, traj_in = self.mr.apply_interaction(
                self.model, d[0], text,  label=False)
            vprint(pred.shape)
            vprint(limits(pred))

            print(pred)
            print(pred.shape)
            print(t0, tf)

            # cnt = np.ones([traj_in.shape[0]])*0.10
            # pred_cnt = self.mr.follow_hard_constraints(
            #     traj_in, pred[0, :, :], cnt)
            new_traj_simple = interpolate_2points(pred[0, :, :], t0, tf)

            # x, y = interpolate_traj(new_traj_simple, n=1000)
            new_traj_wp = fit_wps_to_traj(new_traj_simple, traj_raw)

            self.constraints = np.ones([traj_raw.shape[0]])*0.15

            new_traj_wp_cnt = self.mr.follow_hard_constraints(
                traj_raw, new_traj_wp, self.constraints)

            new_traj_wp_scaled = self.mr.addapt_to_hard_constraints(
                traj_raw, new_traj_wp, self.constraints)

            pred_t = np.transpose(pred[:, :, :2], [0, 2, 1])
            pred_d = pred_t.reshape([pred_t.shape[0], pred_t.shape[2]*2])

            plt.clf()

            show_data(d, pred=pred_d, abs_pred=True,
                      show_label=False, new_fig=False,  show_interpolated=True, show_original=False)
            plt.pause(.001)

            # plt.plot(new_traj_wp_scaled[:, 0],
            #          new_traj_wp_scaled[:, 1], "*-", color="yellow")
            # plt.plot(new_traj_wp_cnt[:, 0],
            #          new_traj_wp_cnt[:, 1], "*-", color="green")
            # plt.plot(new_traj_wp[:, 0], new_traj_wp[:, 1], "*-", color="red")
            # plt.plot(traj_raw[:, 0], traj_raw[:, 1], "*-", color="blue")
            # plt.show(block=False)
            # plt.pause(.000001)
            # update traj
            self.new_traj_no_constraints = self.update_traj(
                self.traj, new_traj_wp, mod=mod)
            # self.new_traj = self.update_traj(
            #     self.traj, new_traj_wp_scaled, mod=mod)

            self.new_traj = self.update_traj(
                self.traj, new_traj_simple, mod=mod)

        rospy.loginfo("New trajectory updated!")

    def update_traj(self, traj, new_traj_wp, mod=1):
        new_traj = []

        for i in range(self.traj_trim_start):
            new_traj.append(self.traj[i].goal)

        print("new_traj_wp", len(new_traj_wp))
        for i in range(len(new_traj_wp)):
            wp = self.traj[i*mod+self.traj_trim_start]
            new_goal = MoveEEToPoseGoal()
            new_goal.halt_on_goal = wp.goal.halt_on_goal
            new_goal.subsequent_pose = wp.goal.subsequent_pose
            # new_goal = wp.goal

            p = dqmsg2pose(wp.goal.target_pose)
            # vprint(p.position.x,p.position.y,p.position.z, "    ", p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)

            p.position.x = new_traj_wp[i, 0]
            p.position.y = new_traj_wp[i, 1]
            dq = pose2dq(p)
            new_goal.target_pose = dq2dqMsg(dq)

            if i+mod < self.traj_trim_end:
                wp_next = traj[self.traj_trim_start+i+mod]
            else:
                wp_next = self.traj[self.traj_trim_end]
                print("new_traj ", i)
            # vprint(wp_next)
            # p = dqmsg2pose(wp_next.goal.target_pose)
            # dq = pose2dq(p)
            # new_goal.subsequent_pose = dq2dqMsg(dq)
            new_goal.subsequent_pose = wp_next.goal.target_pose

            new_traj.append(new_goal)
        for i in range(self.traj_trim_end, len(self.traj)):
            new_traj.append(self.traj[i].goal)
        return new_traj

    def execute_new_traj(self, msg):
        # transmit traj
        self.publish_traj(self.new_traj)

    def go_to_init(self, msg):
        index = 0
        if len(msg.data) > 0:
            index = int(msg.data)
        self.publish_traj([self.new_traj[index]])

    def save_traj(self, file_name='last_traj.json'):

        if not isinstance(file_name, str):
            if len(file_name.data) == 0:
                file_name = 'last_traj.json'
            else:
                file_name = file_name.data
        d = {}
        for i, wp in enumerate(self.traj):

            d[i] = obj2dict(wp)
        with open(file_name, 'w') as f:
            json.dump(d, f)
            rospy.loginfo("saved traj to: "+file_name +
                          ", it has "+str(len(self.traj))+" waypoints")

        return

    def get_traj_from_file(self, file_name='last_traj.json'):
        traj = []
        with open(file_name) as f:
            data = json.load(f)
            self.reset_traj()
            traj = [None]*len(data.keys())

            for k, v in data.items():
                wp = MoveEEToPoseActionGoal()
                dict2obj(v, wp)
                traj[int(k)] = wp
        return traj

    def load_traj(self, file_name='last_traj.json'):
        if not isinstance(file_name, str):
            if len(file_name.data) == 0:
                file_name = 'last_traj.json'
            else:
                file_name = file_name.data
        self.traj = self.get_traj_from_file(file_name=file_name)
        # with open(file_name) as f:
        #     data = json.load(f)
        #     self.reset_traj()
        #     self.traj = [None]*len(data.keys())

        #     for k, v in data.items():
        #         wp = MoveEEToPoseActionGoal()
        #         dict2obj(v, wp)
        #         self.traj[int(k)] = wp
        rospy.loginfo("loaded the traj from: "+file_name +
                      " with "+str(len(self.traj))+" waypoints")

    def callback(self, data):
        if self.record:
            self.traj.append(data)
        # vprint(self.traj)

    def record_traj(self, data):

        self.record = data.data
        rospy.loginfo("traj recording: "+str(self.record))

    # ========== PUBLISHERS and ACTIONS ================

    def connect_to_server(self):
        self.traj_client = actionlib.SimpleActionClient(
            '/goToPose', MoveEEToPoseAction)
        self.traj_client.wait_for_server()

    def publish_traj(self, traj):

        self.traj_sub.unregister()
        rospy.loginfo("unsubscribed from traj topic.")
        rospy.sleep(0.5)
        self.connect_to_server()
        rospy.loginfo("connected to action server")

        rospy.loginfo("publishing at traj topic.")
        for i, goal in enumerate(traj):
            self.traj_client.send_goal(goal)
            self.traj_client.wait_for_result()
        rospy.loginfo("DONE publishing.")

        self.traj_sub = rospy.Subscriber(
            "goToPose/goal", MoveEEToPoseActionGoal, self.callback)

    # =============== UTIL functions ==============

    def reset_traj(self, data=None):
        rospy.loginfo("Trajectory reseted")

        self.traj = []
        self.new_traj = []
        self.new_traj_no_constraints = []


if __name__ == '__main__':

    m = Motion_refiner_wrapper(load_model=False)
    m.load_traj(file_name='/home/gari/wine.json')

    # m.obj_poses = np.array([[1.2673348043972765, 1.3, 1.3],
    #                         [0.0, 0.3, -0.20]])

    # plt.ion()
    # fig = plt.figure()
    # for i in range(5):
    #     m.apply_interaction("keep a bigger distance from the glasses")
    #     m.apply_interaction("keep a much bigger distance from the glasses")
    #     m.apply_interaction("stay a bit further away from the glasses")
    #     m.apply_interaction("stay further away from the glasses")
    #     m.apply_interaction("stay a lot further away from the glasses")
    #     m.obj_poses += np.array([[-0.2, -0.2, -0.2], [0, 0, 0]])

    # m.load_traj(file_name='last_traj.json')

    # traj = wplist2np(m.traj)
    # plt.plot(np.arange(len(traj[1:, 3])), traj[1:, 3]-traj[:-1, 3])

    # plt.show(block=False)
    # plt.figure()
    # plt.plot(np.arange(len(traj[1:, 0])), traj[1:, 0]-traj[:-1, 0])
    # plt.show()
    # mask = np.where(traj[1:, 3]-traj[:-1, 3] > 0.5, True, False)
    # filtered_traj = traj[1:, :][mask, :]

    # plt.plot(filtered_traj[:-1, 0], filtered_traj[:-1, 1])
    # o = 0.2
    # plt.plot(traj[:-1, 0]+o, traj[:-1, 1]+o)

    # plt.show()
    # m.apply_interaction("keep a bigger distance from the glasses")
    # m.apply_interaction("keep a bigger distance from the phone")
    # m.apply_interaction("walk further away from the computer")

    # m.apply_interaction("pass much closer to the wine")

    # m.apply_interaction("drive very further away from the soda soda bottle")

    # m.apply_interaction("stay much further away from the computer")
    # m.apply_interaction("pass really close to the computer")
    # m.apply_interaction("walk much closer to the the cellphone")
    # m.apply_interaction("stay much further away from the cup")
    # m.apply_interaction("stay really close to the the cellphone")

    # m.apply_interaction("stay closer to the bottle")
    # m.apply_interaction("stay further away from the cellphone")

    # plt.plot(X[:,0],X[:,1],X[:,2],'o-')
    # ax = plt.axes(projection='3d')
    # ax.plot3D(X[:,0],X[:,1],X[:,2], 'gray')
    # plt.plot(range(X.shape[0]-1), X[1:, 3]-X[:-1, 3])
    # print(np.max(X[:,3]))
    # print(X[:, 3])

    # plt.show()
    # plt.show(block=False)

    # m.traj=[t,t,t]

    try:
        rospy.spin()
    except:
        pass
    # p1 = Pose()

    # print(getmembers(dqrobotics , isfunction))

    # dq = DQ([-0.477558981437,0.8770795349,-0.049187221985,0.0157963888759,-0.272307093539,-0.144241530109,0.0302323724449,-0.129419305155])
    # print("====> old")
    # print(dq)

    # p = dq2pose(dq)
    # dq_new = pose2dq(p)
    # # dq_msg = dq2dqMsg(dq)
    # print("====> new")
    # print(dq_new)
