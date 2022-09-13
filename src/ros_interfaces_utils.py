from my_interfaces.msg import MoveEEToPoseActionGoal, DualQuaternion, MoveEEToPoseAction, MoveEEToPoseGoal

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