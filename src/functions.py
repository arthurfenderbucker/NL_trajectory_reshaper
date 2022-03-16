import random
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf


# ================= plot functions ====================


def plot_questions(user_data, questions, interaction_name,  file=""):
    fig = plt.figure(figsize=(len(questions)*2, 8))
    for j, q in enumerate(questions):
        ax = plt.subplot(2, 5, j+1)
        ax.set_title(q)
        plt.yticks([], rotation=45, fontsize=10)
        # if j > 0:
        # ax.axis('off')
        # ax = fig.add_axes([0,0,1,1])
        sum_vec = np.array([0, 0, 0, 0])
        for i in range(0, len(user_data)):
            sum_vec += user_data[i][q]
        norm = len(user_data)*5/100
        # plt.xticks(interaction_name,rotation=60, horizontalalignment='right', fontsize=12)
        ax.bar(interaction_name, sum_vec/norm, color="#cc7818")
        plt.xticks(rotation=45, fontsize=10)
        print(q, " norm score", sum_vec/norm)
        print(q, " average:", sum_vec/len(user_data))

    plt.tight_layout()
    if file != "":
        plt.savefig(file)
    plt.show()


def plot_criteria(user_data, criteria, interaction_name, file=""):
    fig = plt.figure(figsize=(len(criteria)*3, 3))
    for j, q in enumerate(criteria):
        ax = plt.subplot(1, 5, j+1)
        ax.set_title(q)
        # ax = fig.add_axes([0,0,1,1])
        sum_vec = np.array([0, 0, 0, 0])
        for i in range(0, len(user_data)):
            vec = [0, 0, 0, 0]
            for id, c in enumerate(user_data[i][q]):
                vec[ord(c)-ord("a")] = 3-id
            sum_vec += vec
        norm = len(user_data)*4/100
        # plt.xticks(interaction_name,rotation=60, horizontalalignment='right', fontsize=12)
        plt.yticks([], rotation=45, fontsize=10)
        plt.xticks(rotation=45, fontsize=10)
        ax.bar(interaction_name, sum_vec/norm, color="#cc7818")
        print(q, " norm score", sum_vec/norm)
        print(q, " average:", sum_vec/len(user_data))
    plt.tight_layout()
    if file != "":
        plt.savefig(file)
    plt.show()


def plot_XY(x, y, n_objs=3, new_fig=True, cx="blue", cy="green", x_lim=[0, 1], y_lim=[0, 1]):
    n = x.shape[0]
    rows = int(np.ceil(n/3))
    if new_fig:
        plt.figure(figsize=(12, rows*3))
        plt.axes().set_aspect('equal')
    

    for i in range(n):

        plt.subplot(int(np.ceil(n/3)), min(n, 3), i+1)
        plt.title(i, fontsize=8)

        plt.plot(x[i, n_objs:, 0], x[i, n_objs:, 1], cx)
        plt.plot(y[i, :, 0], y[i, :, 1], cy)
        plt.scatter(x[i, :n_objs, 0], x[i, :n_objs, 1])
        plt.xlim(x_lim)
        plt.ylim(y_lim)


def show_data(d_,show=True, obj_txt=False,arrows=True,file = "",plot_mse = False, n_col = 5, obj_c ="#363634", grid_c="#cfcfcf",ref_c = "#d10808", label_c = "#3082e6", pred_c = "#2dd100", delta_c = "#e3d64b",fig_mult=3,pred=None,new_fig = True, n=3, abs_pred=False,  show_label=True, show_interpolated=False, show_original=True):

    if isinstance(pred, np.ndarray) and len(pred.shape) > 2:
        print("reshaping")
        pred = pred.reshape([pred.shape[0], pred.shape[1]*2])

    if isinstance(d_, np.ndarray):
        D = d_
    elif not isinstance(d_, list):
        D = [d_]
    else:
        D = d_


    rows = int(np.ceil(len(D)/n_col))
    if new_fig:
        fig = plt.figure(figsize=(fig_mult*5, rows*fig_mult))
        fig.patch.set_facecolor('xkcd:white')
    plt.axes().set_aspect('equal')
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=0.2, wspace=0.001, hspace=0.001)
    
    # print(D[0].shape)
    total_e = 0
    for i, d in enumerate(D):

        ax = plt.subplot(int(np.ceil(len(D)/n_col)), min(len(D), n_col), i+1)
        ax.set_xticklabels(())
        ax.set_yticklabels(())
        ax.set_aspect('equal')
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,  
            left=False,
            right=False, 
            labelbottom=False)

        # ax.grid(b=True, which='major', color=grid_c, linestyle='-',
        #         alpha=0.5, xdata=np.arange(5)/5,ydata=np.arange(5)/5)
        for xi in [0.2,0.4,0.6,0.8]:
            ax.axhline(xi, linestyle='-', color=grid_c,alpha=0.5, linewidth=1) # horizontal lines
            ax.axvline(xi, linestyle='-', color=grid_c,alpha=0.5, linewidth=1) # vertical lines
        ax.set_xlim([0.0,1.0])
        ax.set_ylim([0.0,1.0])
        ax.set_title(d["text"], fontsize=7)

        traj_in = np.array(d["input_traj"])
        if show_original:
            ax.plot(traj_in[:, 0], traj_in[:, 1], ref_c)
        if show_label:
            traj_out = np.array(d["output_traj"])
            ax.plot(traj_out[:, 0], traj_out[:, 1], label_c)

        p = d["obj_poses"]
        if isinstance(p, list):
            p = np.array(p)
        ax.scatter(p[0, :], p[1, :],color=obj_c)

        if not pred is None:
            cut = int(len(pred[i])/2)

            # interpolate the traj
            tck_i, u_i = interpolate.splprep(traj_in.T, s=0.0)
            x_i, y_i = interpolate.splev(np.linspace(0, 1, cut), tck_i)

            if show_interpolated:
                ax.plot(x_i, y_i, "blue")
            if show_label:
                tck_o, u_o = interpolate.splprep(traj_out.T, s=0.0)
                x_o, y_o = interpolate.splev(np.linspace(0, 1, cut), tck_o)

            factor = 1.0
            if abs_pred:
                ax.plot(pred[i, 0:cut]*factor, pred[i, cut:]*factor, pred_c)

            else:
                ax.plot(x_i+pred[i, 0:cut]*factor,
                         y_i+pred[i, cut:]*factor, pred_c)

            if show_label:
                error = np.average((pred[i, 0:cut]-x_o)
                                   ** 2+(pred[i, cut:]-y_o)**2)
                total_e += error
                if plot_mse:
                    ax.text(0.05, 0.85, "MSE:%.1fe-3" % (error*1000.0))

            if arrows:
                for pi in range(cut):
                    if abs_pred:
                        ax.arrow(x_i[pi], y_i[pi],pred[i, pi]-x_i[pi], pred[i, pi+cut]-y_i[pi],head_width = 0.02,width = 0.005,ec =delta_c,color =delta_c,length_includes_head=True)
                        # ax.plot([x_i[pi], pred[i, pi]], [
                        #          y_i[pi], pred[i, pi+cut]], markersize=1, marker='o', color=delta_c )
                    else:
                        ax.arrow(x_i[pi], y_i[pi],pred[i, pi], pred[i, pi+cut],head_width = 0.02,width = 0.005,ec =delta_c,color =delta_c,length_includes_head=True)

                        # ax.plot([x_i[pi], x_i[pi]+pred[i, pi]], [y_i[pi], y_i[pi] +
                        #          pred[i, pi+cut]], markersize=1, marker='o', color=delta_c )

        for j, n in enumerate(d["obj_names"]):
            # ax.text(p[0, j], p[1, j], n)
            if obj_txt:
                ax.text(p[0, j], p[1, j]+0.05,n, bbox={'facecolor':'white','alpha':0.0,'edgecolor':'none','pad':1},
                        ha='center', va='center', fontsize=10) 
        # if i >= n:
        #     break
    plt.tight_layout()
    if show_label:
        print(total_e/len(D))
    if file != "":
        plt.savefig(file)
    if show:
        plt.show()


def plot_loss(history):
    # summarize history for loss
    plt.figure(figsize=(3, 1))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


def plot_dist(x):
    plt.figure(figsize=(3, 1))
    avr = np.average(x, (0))
    e = np.std(x, (0))
    plt.errorbar(np.arange(len(avr[0, :])), avr[0, :],
                 e[0, :], linestyle='None', marker='^')
    plt.errorbar(np.arange(len(avr[1, :]))+0.25, avr[1, :],
                 e[1, :], linestyle='None', marker='^', color="red")

    plt.show()

# ========== data augmentation functions =============


def augment_xy(x, y, width_shift_range=0.1, height_shift_range=0.1, rotation_range=np.pi,
               zoom_range=[0.5, 1.2], horizontal_flip=True, vertical_flip=True, offset=[-0.5, -0.5]):

    r_ang = 0
    ns, nt, nd = x.shape
    R = np.tile(np.eye(2), [ns, 1, 1])
    z_f = np.ones([ns, 1, 1])
    sf = np.zeros([ns, 1, nd])
    flip_f = np.ones([ns, 1, nd])

    if width_shift_range != 0.0:
        sf[:, :, 0] = np.random.uniform(-width_shift_range,
                                        width_shift_range, [ns, 1])
    if height_shift_range != 0.0:
        sf[:, :, 1] = np.random.uniform(-height_shift_range,
                                        height_shift_range, [ns, 1])
    if rotation_range != 0.0:
        r_ang = np.random.uniform(-rotation_range, rotation_range, [ns, 1])
        c = np.cos(r_ang)
        s = np.sin(r_ang)

        l1, l2 = np.concatenate(
            [c, s], axis=-1), np.concatenate([s, -c], axis=-1)
        R = np.concatenate(
            [l1[:, np.newaxis, :], l2[:, np.newaxis, :]], axis=-2)

    if zoom_range != 0.0:
        z_f = np.random.uniform(zoom_range[0], zoom_range[1], [ns, 1, 1])

    if vertical_flip:
        flip_f[:, :, 0] = np.random.choice([-1, 1], [ns, 1])
    if horizontal_flip:
        flip_f[:, :, 1] = np.random.choice([-1, 1], [ns, 1])

    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)

    for s in range(ns):
        x_new[s] = (np.dot((x[s, :, :]-0.5)*flip_f[s, :, :],
                           R[s, :, :])+0.5)*z_f[s, 0, 0]+sf[s, :, :]+offset[1]
        y_new[s] = (np.dot((y[s, :, :]-0.5)*flip_f[s, :, :],
                           R[s, :, :])+0.5)*z_f[s, 0, 0]+sf[s, :, :]+offset[1]

    return x_new, y_new


# def plot_attention(attention, sentence, predicted_sentence):
#     sentence = tf_lower_and_split_punct(sentence).numpy().decode().split()
#     predicted_sentence = predicted_sentence.numpy().decode().split() + ['[END]']
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(1, 1, 1)

#     attention = attention[:len(predicted_sentence), :len(sentence)]

#     ax.matshow(attention, cmap='viridis', vmin=0.0)

#     fontdict = {'fontsize': 14}

#     ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
#     ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     ax.set_xlabel('Input text')
#     ax.set_ylabel('Output text')
#     plt.suptitle('Attention weights')


# ============= trajectories functions ===============

def np2data(input_traj, obj_names, obj_poses, text, output_traj=None):
    data = []
    data.append({"input_traj": input_traj, "output_traj": output_traj, "text": text, "obj_names": obj_names,
                 "obj_poses": obj_poses})
    return data


def filter(x, y, data, lower_limit=-0.98, upper_limit=0.98):
    # filters samples where the out traj is not between 0 and 1
    i_invalid = np.concatenate([np.arange(len(y))[np.min(
        y, axis=1) < lower_limit], np.arange(len(y))[np.max(y, axis=1) > upper_limit]])
    y_filtered = np.delete(y, i_invalid, axis=0)
    X_filtered = np.delete(x, i_invalid, axis=0)
    i_valid = np.delete(np.arange(len(y)), i_invalid, axis=0)
    data_filtered = [data[i] for i in i_valid]
    return X_filtered, y_filtered, data_filtered, i_invalid

def arg_max_obj(x, data, obj_sim_indices, words=["left","right","front","back"]):
    # filters samples where the out traj is not between 0 and 1
    i_no_obj = np.array([i for i,d in enumerate(data) if has_word(d["text"], words) ],dtype=np.int32)

    a = x[:,obj_sim_indices]
    mask = np.zeros_like(a)
    mask[np.arange(a.shape[0]),np.argmax(a,axis=1)] = 1
    mask[i_no_obj,:] = 0
    
    x[:,obj_sim_indices] = mask
    return x, i_no_obj

def has_word(t, words):
    for w in words:
        if w in t:
            return True
    return False


def filter_cartesian(x, y, data, words=["left", "right", "front", "back"]):
    # filters samples where the out traj is not between 0 and 1

    i_invalid = np.array([i for i, d in enumerate(data)
                          if has_word(d["text"], words)])
    y_filtered = np.delete(y, i_invalid, axis=0)
    X_filtered = np.delete(x, i_invalid, axis=0)
    i_valid = np.delete(np.arange(len(y)), i_invalid, axis=0)
    data_filtered = [data[i] for i in i_valid]
    return X_filtered, y_filtered, data_filtered, i_invalid


def limits(x):
    return x.shape, np.min(x), np.max(x)


def reshape_input(y, d=2):
    out = np.transpose(
        y.reshape([y.shape[0], 2, int(y.shape[1]/2)]), [0, 2, 1])
    return out


def reorganize_input(y, d=2):
    traj_n = int(y.shape[-1]/d)
    out = np.zeros_like(y)
    for i in range(d):
        out[:, np.arange(i, traj_n*d, d)] = y[:,
                                              np.arange(i*traj_n, (i+1)*traj_n)]
    return out


def list_to_wp_seq(y):
    out = np.transpose(
        y.reshape([y.shape[0], 2, int(y.shape[1]/2)]), [0, 2, 1])
    return out


def tokenize(y, n_classes=1002):
    # out = np.transpose(y.reshape([y.shape[0],2,int(y.shape[1]/2)]),[0,2,1])
    out = reorganize_input(y)
    out = out*(n_classes-2)+1
    # out = np.concatenate([np.zeros([out.shape[0],1,2]),out,np.ones((out.shape[0],1,2))*(n_classes-1)],axis=1)
    return out.astype('int32')


def generate(model, source, traj_n=10):
    """Performs inference over one batch of inputs using greedy decoding."""
    traj, shifted_target, features = source
    bs = tf.shape(traj)[0]
    dec_input = traj[:, 3:4, :]  # tf.ones((bs, 1,2)) * init_dec_input
    for i in range(traj_n - 1):
        dec_out = model.predict([traj, dec_input, features])
        dec_input = tf.concat([dec_input, dec_out[:, -1:, :]], axis=1)
    return dec_input


# ============ dataset manipulation functions ==================

def reset_logs(logdir):
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


if __name__ == '__main__':
    pass