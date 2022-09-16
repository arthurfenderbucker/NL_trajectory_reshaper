import random
from tabnanny import verbose
from matplotlib import projections
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import os
import similaritymeasures

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

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


def show_data(d_,show=True, obj_txt=False,arrows=True,file = "",plot_mse = False, n_col = 5,obj_c ="#363634", grid_c="#cfcfcf",ref_c = "#d10808", label_c = "#3082e6", pred_c = "#2dd100", delta_c = "#e3d64b",fig_mult=3,pred=None,new_fig = True, n=3, abs_pred=False,  show_label=True, show_interpolated=False, show_original=True):

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






# =============== plot 3D and 4D functions =====================


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)



def show_data4D(d_,image_loader= None,pred=None, show=True,color_traj=True, obj_txt=False,arrows=True,file = "", cmap=None,
        n_col = 2, obj_c ="#363634", grid_c="#cfcfcf",ref_c = "#d10808", label_c = "#3082e6", pred_c = "#2dd100", delta_c = "#e3d64b",
        fig_mult=3,new_fig = True, n=3, abs_pred=False,  show_label=True, show_interpolated=False, show_original=True,
        change_img_base=None, plot_forces=False,plot_output=True):

    # if isinstance(pred, np.ndarray) and len(pred.shape) > 2:
    #     print("reshaping")
    #     pred = pred.reshape([pred.shape[0], -1, 4])

    if isinstance(d_, np.ndarray):
        D = d_
    elif not isinstance(d_, list):
        D = [d_]
    else:
        D = d_

    if cmap is None:
        cmap=plt.cm.plasma
    
    # rows = int(np.ceil(len(D)/n_col))
    # fig = plt.figure(constrained_layout=True, figsize=(fig_mult*5, rows*fig_mult))
    # print(type(fig))
    # subfigs = fig.subfigures(int(np.ceil(len(D)/n_col)), min(len(D), n_col), wspace=0.07)
    # print(type(subfigs),len(subfigs))
    # if new_fig:
    #     fig = plt.figure(figsize=(fig_mult*5, rows*fig_mult))
    #     fig.patch.set_facecolor('xkcd:white')

    # plt.axes().set_aspect('equal')
    # plt.tight_layout()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=0.2, wspace=0.001, hspace=0.001)
    
    # print(D[0].shape)
    total_e = 0
    for i, d in enumerate(D):

        pts = np.asarray(d["input_traj"])
        pts_new = np.asarray(d["output_traj"])
        text = d["text"]
        obj_names = np.asarray(d["obj_names"])
        obj_pt = np.asarray(d["obj_poses"])
        # obj_classes = np.asarray(d["obj_classes"])
        # obj = d["obj_in_text"]
        # lg_ct = d["change_type"]
        # mi = d["map_id"]
        image_paths = d["image_paths"]
        forces=None
        if "forces" in d.keys():
            forces = d["forces"]

        if not change_img_base is None:
            for ti in range(len(image_paths)):
                image_paths[ti] = image_paths[ti].replace(change_img_base[0], change_img_base[1])

        objs  = {}
        for x,y,z,name in zip(obj_pt[:,0],obj_pt[:,1],obj_pt[:,2],obj_names):
            objs[name] = {"value":{"obj_p":[x,y,z]}}

        new_pts_list = [pts_new]
        if d["output_traj"] is None or not plot_output:
            new_pts_list = []

        if not pred is None:
            new_pts_list.append(pred[i])
        # print(new_pts_list)
        # images_path = "/home/mirmi/Arthur/dataset/"
        # objs_img_base_paths = [images_path+c+"/"+n for c,n in zip(obj_classes,obj_names)]

        # obj_img_paths = [im_path+"/"+random.choice(os.listdir(im_path)) for im_path in objs_img_base_paths]
        objs_images = []
        if not image_loader is None:
            objs_images = [image_loader(im) for im in image_paths]

        plot_samples(text,pts,new_pts_list, images=objs_images,objs=objs, color_traj =color_traj,forces=forces)


        # if color_traj:
        #     norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
        #     fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.026, pad=0.04,label='speed change')

    if file != "":
        plt.savefig(file)
    if show:
        plt.show()

def plot3Dcolor(x, y, z, c, ax=None, cmap=None, **args):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    if cmap is None:
        cmap=plt.cm.plasma
        
    for i in range(1, len(x)):
        ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], c=cmap(c[i-1]), **args)
    
    ax.scatter(x[-1:], y[-1:], z[-1:],s=np.ones_like(z[:1])*100,marker="D",color=cmap(c[-1])) #alpha sets the darkness of the path.

    

def plot_samples(text,pts,pts_new_list, images=[], fig=None,objs=None, colors = ["#0071b3", "#1e0191"],alpha=[0.9,0.9],
                plot_voxels= False, color_traj = False, map_cost_f=None, labels=[], plot_speed=True, show=True, forces=None):
    
    start_color = "red"
    if len(labels) == 0:
        labels = [text]*len(pts_new_list)

    if fig is None:
        fig = plt.figure(figsize=(10,13))

        fig.add_subplot(1,1,1,projection='3d')
    ax = plt.gca(projection='3d')

    cmap=plt.cm.viridis
    # plot3Dcolor(x_init, y_init, z_init, vel_init,ax=ax, cmap=cmap,linewidth=5.0)
    
    x_init, y_init, z_init, vel_init = pts[:,0],pts[:,1],pts[:,2], pts[:,3]
    line, = ax.plot(x_init, y_init, z_init,alpha=0.9,color="red", label="ORIGINAL") 
    if color_traj:
        norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.026, pad=0.04,label='speed change')


    if plot_speed:
        ax2 = fig.add_subplot(9,2,18)
        ax2.plot(np.arange(len(vel_init)),vel_init+1,color="red",label="original")

    for i, pts_new in enumerate(pts_new_list):
        x_new, y_new, z_new, vel_new = pts_new[:,0],pts_new[:,1],pts_new[:,2],pts_new[:,3]

        color = colors[i] if i < len(colors)-1 else "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])    
        alpha_ = alpha[i+1] if i+1 < len(alpha)-1 else 0.9    

        if color_traj:
            plot3Dcolor(x_new, y_new, z_new, vel_new-vel_init+0.5,ax=ax, cmap=cmap,linewidth=5.0)
            
        else:
            ax.plot(x_new, y_new, z_new,alpha=alpha_, color=color, label=labels[i]) 
        
        
        if plot_speed:
            ax2.plot(np.arange(len(vel_init)),vel_new+1,color=color, label=labels[i])
            ax2.set_xlabel('waypoints')
            ax2.set_ylabel('')
            ax2.set_title("speed profile")
        
        # ax.scatter(x_new[:-1], y_new[:-1], z_new[:-1],alpha=0.9,color=color) 

        if plot_voxels and not map_cost_f is None:
            grid = np.mgrid[0:1:0.1,0:1:0.1,0:1:0.1]
            x,y,z = grid

            cost = np.linalg.norm(map_cost_f(grid[:,:-1,:-1,:-1].T.reshape((-1,3))),axis=1)
            
            data = cost.reshape([9,9,9]).T*2.0
            
            visiblebox = np.random.choice([True,False],data.shape)
            visiblebox = np.where(data>0.0, True, False)
            colors = plt.cm.plasma(data)
            norm = matplotlib.colors.Normalize(vmin=np.min(data), vmax=np.max(data))
            vox = ax.voxels(x,y,z,visiblebox,facecolors=colors,alpha = 0.1,edgecolor='None')

            m = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
            m.set_array([])
            plt.colorbar(m)

    if not forces is None:
        for i, (pt, f) in enumerate(zip(pts, forces)):
            
            s=10.0
            a = Arrow3D([pt[0], pt[0]+f[0]*s], [pt[1], pt[1]+f[1]*s], 
                    [pt[2], pt[2]+f[2]*s], mutation_scale=5, 
                    lw=1, arrowstyle="-|>", color="b")
            ax.add_artist(a)


    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(text)
    if not objs is None:
        for name,v in objs.items():
            x,y,z = v["value"]["obj_p"]
            ax.scatter(x,y,z)
            ax.text(x, y, z, name, 'x')
    
    set_axes_equal(ax)
    img_ax = []
    # im1 = np.arange(100).reshape((10, 10))
    # im2 = im1.T
    # im3 = np.flipud(im1)
    # im4 = np.fliplr(im2)
    # images = [im1, im2, im3, im4]
    if len(images)>0:
        grid = ImageGrid(fig, (7,2,13),  # similar to subplot(111)
                    nrows_ncols=(1, len(images)),  # creates 2x2 grid of axes
                    axes_pad=0.02,  # pad between axes in inch.
                    )

        for obj_name,ax_im, im in zip(objs.keys(),grid, images):
            # Iterating over the grid returns the Axes.
            ax_im.imshow(im)
            ax_im.axes.xaxis.set_ticklabels([])
            ax_im.axes.yaxis.set_ticklabels([])
            ax_im.set_title(obj_name,fontsize=8)

    # for i,img in enumerate(images):
    #     img_ax.append(fig.add_subplot(7, 7, i))
    #     img_ax[i]plt.imshow(img)
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1])

    if plot_speed:
        handles, labels = ax2.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])
    else:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1])

    if show:
        plt.show()
    return fig


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])




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




def augment_4D(x, y,shift_range=[],
               zoom_range=[0.5, 1.2], flip=False, offset=[]):

    r_ang = 0
    ns, nt, nd = x.shape
    R = np.tile(np.eye(2), [ns, 1, 1])
    z_f = np.ones([ns, 1, 1])
    sf = np.zeros([ns, 1, nd])
    flip_f = np.ones([ns, 1, nd])
    offset_f = np.zeros([ns, 1, nd])

    if len(shift_range)>0:
        for i in range(len(shift_range)):
            sf[:, :, i] = np.random.uniform(-shift_range[i],
                                        shift_range[i], [ns, 1])
    if zoom_range != 0.0:
        z_f = np.random.uniform(zoom_range[0], zoom_range[1], [ns, 1, 1])

    if flip:
        for i in nd:
            flip_f[:, :, i] = np.random.choice([-1, 1], [ns, 1])
    if len(offset)>0:
        for i in range(len(offset)):
            offset_f[:,:,i] = offset[i]

    x_new = (x*flip_f[:, :, :])*z_f[:, :, :]+sf[:, :, :]+offset_f[:,:,:]
    y_new = (y*flip_f[:, :, :])*z_f[:, :, :]+sf[:, :, :]+offset_f[:,:,:]

    # y_new[s] = 
    # for s in range(ns):
    #     x_new[s] = (np.dot((x[s, :, :]-0.5)*flip_f[s, :, :],
    #                        R[s, :, :])+0.5)*z_f[s, 0, 0]+sf[s, :, :]+offset[1]
    #     y_new[s] = (np.dot((y[s, :, :]-0.5)*flip_f[s, :, :],
    #                        R[s, :, :])+0.5)*z_f[s, 0, 0]+sf[s, :, :]+offset[1]

    return x_new, y_new


# ================ metrics =============================


def compute_metrics(trajs_x, trajs_y, filter_nan=False):
    """Computes the similarity metrics between 2 trajectories:
    returns: dict(pcm,df,area,cl,dtw,mae,mse), metrics over each sample"""

    metrics = {"pcm":None,"dfd":None,"area":None,"cl":None,"dtw":None,"mae":None,"mse":None}

    metrics_h = np.zeros((trajs_x.shape[0],7))
    for i,(exp_data, num_data) in enumerate(zip(trajs_x, trajs_y)):

        pcm = 0# similaritymeasures.pcm(exp_data, num_data) # Partial Curve Mapping
        dfd = similaritymeasures.frechet_dist(exp_data, num_data) # Discrete Frechet distance
        area = 0 #similaritymeasures.area_between_two_curves(exp_data, num_data) # area between two curves
        cl = 0 #similaritymeasures.curve_length_measure(exp_data, num_data)# Curve Length based similarity measure
        dtw, d = similaritymeasures.dtw(exp_data, num_data) # Dynamic Time Warping distance
        mae = similaritymeasures.mae(exp_data, num_data) # mean absolute error
        mse = similaritymeasures.mse(exp_data, num_data) # mean squared error

        metrics_h[i,:] = [pcm,dfd,area,cl,dtw,mae,mse]

    if filter_nan:
        metrics_h = metrics_h[~np.isnan(metrics_h).any(axis=1),:]
    metrics_v = np.mean(metrics_h,axis=0)
    for k,v in zip(metrics.keys(), metrics_v):
        metrics[k]=v
        print(k+":\t",v)
    return metrics, metrics_h

# metrics, metrics_h = compute_metrics(trajs_x, trajs_y)



# ============= trajectories functions ===============


def incorporate_speed(wps, dt=0.1, N=100,speed_offset=1):
    
    #create spline function
    wp_diff = wps[1:,:3]-wps[:-1,:3]
    len_diff = np.linalg.norm(wp_diff, axis=1)
    len_total = np.sum(len_diff)

    # t_total = dt*wps.shape[0]

    speed= wps[:,3]+speed_offset
    dist_covered = 0
    dist_profile = [0]
    time_covered = 0
    new_wps=wps[:1,:3]
    i = 0
    for j,t in enumerate(range(N-1)):
        # print(i)
        step_d = dt*speed[i]
        # print(len_total)
        if step_d+dist_profile[-1] < len_total:
            # print(step_d+dist_profile[-1], dist_covered+len_diff[i])
            while step_d+dist_profile[-1] > dist_covered+len_diff[i]:
                dist_covered += len_diff[i]
                i+=1
            
            new_wp = wps[i,:3]+(step_d+dist_profile[-1]-dist_covered)*wp_diff[i]
        else:
            # print(j, "DONE")
            new_wp = wps[-1,:3]
        new_wps = np.append(new_wps, [new_wp],axis=0)            
        dist_profile.append(step_d+dist_profile[-1])
    return new_wps

def prepare_x(x, traj_indices, obj_poses_indices):
  objs = pad_array(list_to_wp_seq(x[:,obj_poses_indices],d=3),4,axis=-1) # no speed
  trajs = list_to_wp_seq(x[:,traj_indices],d=4)
  return np.concatenate([objs,trajs],axis = 1)


def np2data(input_traj, obj_names, obj_poses, text, output_traj=None, locality_factor=0.5, image_paths=None):
    data = []
    data.append({"input_traj": input_traj, "output_traj": output_traj, "text": text, "obj_names": obj_names,
                 "obj_poses": obj_poses, "locality_factor":locality_factor, "image_paths":image_paths})
    return data


def filter(x, y, data, lower_limit=-0.98, upper_limit=0.98):
    # filters samples where the out traj is not between 0 and 1
    i_invalid = np.concatenate([np.arange(len(y))[np.min(
        y, axis=1) < lower_limit], np.arange(len(y))[np.max(y, axis=1) > upper_limit], np.arange(len(y))[np.any(np.isnan(y),axis=1)]])
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

def pad_array(x,n,v=0,axis = 0): 
    s =list(x.shape)
    s[axis]=n-s[axis]
    return np.concatenate([x,np.ones(s)*v],axis = axis)

# def list_to_wp_seq(y):
#     out = np.transpose(
#         y.reshape([y.shape[0], 2, int(y.shape[1]/2)]), [0, 2, 1])
#     return out

def list_to_wp_seq(y,d=2):
    out = np.transpose(
        y.reshape([y.shape[0], d, int(y.shape[1]/d)]), [0, 2, 1])
    return out


def tokenize(y, n_classes=1002):
    # out = np.transpose(y.reshape([y.shape[0],2,int(y.shape[1]/2)]),[0,2,1])
    out = reorganize_input(y)
    out = out*(n_classes-2)+1
    # out = np.concatenate([np.zeros([out.shape[0],1,2]),out,np.ones((out.shape[0],1,2))*(n_classes-1)],axis=1)
    return out.astype('int32')

def generate_2D(model, source, traj_n=10):
    """Performs inference over one batch of inputs using greedy decoding."""
    traj, shifted_target, features = source
    bs = tf.shape(traj)[0]
    dec_input = traj[:, 3:4, :]  # tf.ones((bs, 1,2)) * init_dec_input
    for i in range(traj_n - 1):
        dec_out = model.predict([traj, dec_input, features], verbose=0)
        dec_input = tf.concat([dec_input, dec_out[:, -1:, :]], axis=1)
    return dec_input


def generate(model, source, traj_n=10, start_index=6):
    """Performs inference over one batch of inputs using greedy decoding."""
    traj, shifted_target, features = source
    bs = tf.shape(traj)[0]
    dec_input = traj[:, start_index:start_index+1, :]  # tf.ones((bs, 1,2)) * init_dec_input
    for i in tqdm(range(traj_n - 1)):
        dec_out = model.predict([traj, dec_input, features], verbose=0)
        dec_input = tf.concat([dec_input, dec_out[:, -1:, :]], axis=1)
    return dec_input

def generator(data_set,stop=False,augment=False, num_objs = 6, pose_on_features=False):

    while True:
        for x, y,emb in data_set:
            x_new, y_new,emb_new = x,y,emb
            if augment:
                x_new, y_new = augment_4D(x,y,shift_range=[0.2,0.2,0.2],zoom_range=[0.6,1.2],flip=False)
                if pose_on_features:
                    emb_new[-num_objs*3:]  = x_new[:num_objs,:3].flatten() # update the obj poses on the feature vector
            yield ( [x_new , y_new[:, :-1],emb_new] , y_new[:, 1:] )
        if stop:
            break



# ============ dataset manipulation functions ==================

def reset_logs(logdir):
    try:
        shutil.rmtree(logdir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def reset_seed(seed = 42):
    tf.random.set_seed(seed)
    try: # versions of Tensorflow
        tf.keras.utils.set_random_seed(seed)
    except:
        pass
    try:
        tf.set_random_seed(seed)
    except:
        pass
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_global_generator(tf.random.Generator.from_seed(seed))
    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    # K.set_session(sess)

if __name__ == '__main__':
    pass