try:
    from my_interfaces.msg import MoveEEToPoseActionGoal, DualQuaternion, MoveEEToPoseAction, MoveEEToPoseGoal
except:
    pass
from geometry_msgs.msg import Pose
from dqrobotics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import sys


def rotation_matrix_from_vectors(vec1, vec2):

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 /
                                                      np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def interpolate_2points(traj, p1, p2, objs=None):
    """traj_new = R*((traj-t1)/n1)*n2 + t2"""

    t1 = traj[0, :]
    t2 = p1

    traj_ = traj-t1
    vec1 = np.concatenate([traj_[-1, :], [0]])
    vec2 = np.concatenate([p2-t2, [0]])
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)

    # print(vec1, vec2)

    R = rotation_matrix_from_vectors(vec1, vec2)[:2, :2]
    # print(R)
    new_traj = np.dot(traj_/n1, R.T)*n2+t2

    if not objs is None:
        new_objs = np.dot((objs-t1)/n1, R.T)*n2+t2
        return new_traj, new_objs
    else:
        return new_traj


def fit_wps_to_traj(t, ref_wps=None, n=10, epslon=0.000001):

    dists = np.linalg.norm(t[:-1, :]-t[1:, :], axis=1)

    total_dist = np.sum(dists)

    if ref_wps is None:
        dists_2 = np.linspace(0, 1, n)
    else:
        if len(ref_wps.shape) == 1:
            dists_2 = ref_wps[:-1]-ref_wps[1:]
        else:
            dists_2 = np.linalg.norm(ref_wps[:-1, :]-ref_wps[1:, :], axis=1)
    total_dist_2 = np.sum(dists_2)
    dists_2 = dists_2*total_dist/total_dist_2

    current_d = 0
    j = 0
    goal_d = 0
    traj_new = np.zeros([len(dists_2)+1, t.shape[-1]])
    traj_new[0, :] = t[0, :]
    traj_new[-1, :] = t[-1, :]
    for i in range(0, len(dists_2)-1):
        goal_d += np.abs(dists_2[i])

        # print(">>>", current_d, goal_d)
        if current_d + np.abs(dists[j]) < goal_d+np.abs(dists_2[i+1]):
            while current_d + np.abs(dists[j]) <= goal_d-epslon:
                current_d += np.abs(dists[j])
                j += 1
                # print("    ", j, current_d, goal_d)
        traj_new[i+1, :] = t[j, :] + \
            (t[j+1, :]-t[j, :])*(goal_d-current_d)/dists[j]
        # print(traj_new[i+1, :])
    return traj_new


def dqmsg2pose(msg):
    dq = DQ([msg.p_r, msg.p_i, msg.p_j, msg.p_k,
            msg.d_r, msg.d_i, msg.d_j, msg.d_k])
    return dq2pose(dq)


def dq2pose(dq):
    m = dq.normalize()
    t = m.translation().vec4()
    # print("t")
    # print(t)

    pose = Pose()
    pose.position.x = t[1]
    pose.position.y = t[2]
    pose.position.z = t[3]

    r = m.vec8()
    pose.orientation.w = r[0]
    pose.orientation.x = r[1]
    pose.orientation.y = r[2]
    pose.orientation.z = r[3]

    return pose


def path2npcoords(path):
    received_traj = []
    for pose in path.poses:
        received_traj.append([pose.pose.position.x, pose.pose.position.y])
    received_traj_np = np.array(received_traj)
    return received_traj_np


def dict2dqMsg(d):
    out = DualQuaternion()
    out.p_r = d["p_r"]
    out.p_i = d["p_i"]
    out.p_j = d["p_j"]
    out.p_k = d["p_k"]
    out.d_r = d["d_r"]
    out.d_i = d["d_i"]
    out.d_j = d["d_j"]
    out.d_k = d["d_k"]
    return out


def dq2dqMsg(dq):
    out = DualQuaternion()
    tmp = dq.vec8()
    out.p_r = tmp[0]
    out.p_i = tmp[1]
    out.p_j = tmp[2]
    out.p_k = tmp[3]
    out.d_r = tmp[4]
    out.d_i = tmp[5]
    out.d_j = tmp[6]
    out.d_k = tmp[7]
    return out


def pose2dq(pose):

    t = DQ(0, pose.position.x, pose.position.y, pose.position.z, 0, 0, 0, 0)
    r = DQ(pose.orientation.w, pose.orientation.x,
           pose.orientation.y, pose.orientation.z, 0, 0, 0, 0)

    return r + 0.5 * r.E * t * r


def obj2dict(obj):
    d = {}
    if(hasattr(obj, '__slots__')):
        for s in obj.__slots__:
            d[s] = obj2dict(getattr(obj, s))
    else:
        d = obj
    return d


def dict2obj(d, obj):
    if isinstance(d, dict):
        for k, v in d.items():
            sub_obj = dict2obj(v, getattr(obj, k))
            setattr(obj, k, sub_obj)
    else:
        obj = d
    return obj
# init_t  = rospy.Time.now()


def wplist2np(wplist):

    out = np.empty((len(wplist), 4))

    for i, wp in enumerate(wplist):
        p = dqmsg2pose(wp.goal.target_pose)
        # t = rospy.Time(wp.goal_id.stamp.secs,wp.goal_id.stamp.nsecs)-init_t
        t = (wp.goal_id.stamp.secs-1644597182) + wp.goal_id.stamp.nsecs/10e9
        out[i, :] = [p.position.x, p.position.y, p.position.z, t]
    return out


if __name__ == '__main__':

    n = 10
    tx = np.arange(n) + np.random.random([n])
    ty = np.arange(n) + np.random.random([n])*10
    t = np.concatenate(
        [tx[:, np.newaxis], ty[:, np.newaxis]], axis=1) * -0.1-0.5

    plt.plot(t[:, 0], t[:, 1])

    p1 = np.array([0.1, 0.1])
    p2 = np.array([0.9, 0.9])

    print(t)

    t_new = interpolate_2points(t, p1, p2)
    plt.plot(t_new[:, 0], t_new[:, 1], "blue")

    t_new = interpolate_2points(t_new, t[0, :], t[-1, :])
    plt.plot(t_new[:, 0], t_new[:, 1], "red")

    print(t_new)

    plt.show()
