import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

import math
import numpy as np

import torchvision

from utils import plotting as utils_plt
from utils import skeleton as util_skel
import torch

from IPython import embed as Ie

def normalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return (pose_tensor-pose_mean)/pose_std

def denormalize_mean_std_tensor(pose_tensor, label_dict):
    pose_mean = label_dict["pose_mean"]
    pose_std  = label_dict["pose_std"]
    return pose_tensor*pose_std + pose_mean

def accumulate_heat_channels(heat_map_batch):
    plot_heat = heat_map_batch[:,-3:,:,:]
    num_tripels = heat_map_batch.size()[1]//3
    for i in range(0, num_tripels):
        plot_heat = torch.max(plot_heat, heat_map_batch[:,i*3:(i+1)*3,:,:])
    return plot_heat

def tensor_imshow(ax, img):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    npimg = np.clip(npimg, 0., 1.)
    ax.imshow(npimg)
    
def tensor_imshow_normalized(ax, img, mean=None, stdDev=None, im_plot_handle=None, x_label=None, clip=True):
    npimg = img.numpy()
    npimg = np.swapaxes(npimg, 0, 2)
    npimg = np.swapaxes(npimg, 0, 1)

    if mean is None:
        mean = (0.0, 0.0, 0.0)
    mean = np.array(mean)
    if stdDev is None:
        stdDev = np.array([1.0, 1.0, 1.0])
    stdDev = np.array(stdDev)

    npimg = npimg * stdDev + mean  # unnormalize
    
    if clip:
        npimg = np.clip(npimg, 0, 1)

    if im_plot_handle is not None:
        im_plot_handle.set_array(npimg)
    else:
        im_plot_handle = ax.imshow(npimg)
        
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    # when plotting 2D keypoints on top, this ensures that it only plots on the image region
    ax.set_ylim([img.size()[1],0])

    if x_label is not None:
        plt.xlabel(x_label)   

    return im_plot_handle

def plot_2Dpose_batch(ax, batch, offset_factor=0.8, bones=util_skel.bones_h36m, colormap='hsv'):
    num_batches = batch.shape[0]
    pose_2d_batchlinear = batch.reshape((num_batches,-1))
    num_joints  = pose_2d_batchlinear.shape[1]//2
    num_bones  = len(bones)
    pose_2d_cat = batch.reshape((-1,2))

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batches):
        # offset bones
        bones_new = []
        offset_i = batchi*num_joints
        for bone in bones:
            bone_new = [bone[0]+offset_i, bone[1]+offset_i]
            if pose_2d_cat[bone_new[0],0] <=0 or pose_2d_cat[bone_new[0],1]<=0 or pose_2d_cat[bone_new[1],0] <=0 or pose_2d_cat[bone_new[1],1] <=0:
                bone_new = [offset_i,offset_i] # disable line drawing, but don't remove to not disturb color ordering
            bones_new.append(bone_new)

        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        offset_x = offset_factor*(batchi %8)
        offset_y = offset_factor*(batchi//8)
        pose_2d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y]])
    #plot_2Dpose(ax, pose_2d, bones, bones_dashed=[], bones_dashdot=[], color='red', linewidth=1, limits=None):
    utils_plt.plot_2Dpose(ax, pose_2d_cat.T, bones=bones_cat, colormap=colormap, color_order=color_order_cat)

def plot_3Dpose_batch(ax, batch_raw, offset_factor_x=None, offset_factor_y=None, bones=util_skel.bones_h36m, radius=0.01, colormap='hsv', row_length=8):
    num_batch_indices = batch_raw.shape[0]
    batch = batch_raw.reshape(num_batch_indices, -1)
    num_joints  = batch.shape[1]//3
    num_bones  = len(bones)
    pose_3d_cat = batch.reshape((-1,3))

    bones_cat = []
    color_order_cat = []
    for batchi in range(0,num_batch_indices):
        # offset bones
        bones_new = []
        offset = batchi*num_joints
        for bone in bones:
            bones_new.append([bone[0]+offset, bone[1]+offset])
        bones_cat.extend(bones_new)
        # offset colors
        color_order_cat.extend(range(0,num_bones))
        # offset skeletons horizontally
        if offset_factor_x is None:
            max_val_x = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            min_val_x = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),0])
            offset_factor_x = 1.5 * (max_val_x-min_val_x)
            radius = offset_factor_x/50
        if offset_factor_y is None:
            max_val_y = np.max(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            min_val_y = np.min(pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),1])
            offset_factor_y = 1.3 * (max_val_y-min_val_y)
        offset_x = offset_factor_x*(batchi % row_length)
        offset_y = offset_factor_y*(batchi // row_length)
        pose_3d_cat[num_joints*batchi:num_joints*(batchi+1),:] += np.array([[offset_x,offset_y,0]])
    utils_plt.plot_3Dpose(ax, pose_3d_cat.T, bones_cat, radius=radius, colormap=colormap, color_order=color_order_cat, transparentBG=True)
    
def plotTransformerBatch(ax_img, transformation, width, height):
    batch_size = transformation.size()[1]
    num_transformers = transformation.size()[0]
    colormap = 'Set1' # cmap = 'hsv'
    clist = ['red','green','blue','orange','cyan','magenta','black','white']

    for i in range(batch_size):
        for j in range(num_transformers):
            affine_matrix = transformation[j, i]
            x_scale = affine_matrix[0, 0].item()
            y_scale = affine_matrix[1, 1].item()
            x_relative = affine_matrix[0, 2].item()
            y_relative = affine_matrix[1, 2].item()
            xwindow = i % 8
            ywindow = i // 8
            #cindex = j*cmap.N // num_transformers
            #color = cmap(cindex)
            color = clist[j]
            rect = patches.Rectangle(
                (width  * (xwindow + (x_relative + 1 - x_scale) / 2),
                 height * (ywindow + (y_relative + 1 - y_scale) / 2)),
                x_scale * width, y_scale * height,
                linewidth=0.2, linestyle='dashed', edgecolor=color, facecolor='none')
            ax_img.add_patch(rect)

def plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name):
    print("labels_raw.keys() = {}, inputs_raw.keys() = {}, outputs_dict.keys() = {}".format(labels_raw.keys(), inputs_raw.keys(), outputs_dict.keys()))
        
    # init figure grid dimensions in an recursive call
    created_sub_plots = 0
    if not hasattr(plot_iol, 'created_sub_plots_last'):
        plot_iol.created_sub_plots_last = {}
    if keyword not in plot_iol.created_sub_plots_last:
        plot_iol.created_sub_plots_last[keyword] = 100 # some large defaul value to fit all..
        # call recursively once, to determine number of subplots
        plot_iol(inputs_raw, labels_raw, outputs_dict, config_dict, keyword, image_name)
        
    num_subplots_columns = 2
    title_font_size = 1
    num_subplots_rows = math.ceil(plot_iol.created_sub_plots_last[keyword]/2)
    
    # create figure
    plt.close("all")
    verbose = False
    if verbose:
        plt.switch_backend('Qt5Agg')
    else:
        plt.switch_backend('Agg') # works without xwindow
    fig    = plt.figure(0)
    plt.clf()

    ############### inputs ################
    # display input images
    for key in inputs_raw.keys():
        if key in ['img','img_crop','bg_crop','bg']:
            images_fg  = inputs_raw[key].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
            ax_img.set_title("Input images", size=title_font_size)
            grid_t = torchvision.utils.make_grid(images_fg, padding=0)
            if 'frame_info' in labels_raw.keys() and len(images_fg)<8:
                frame_info = labels_raw['frame_info'].data
                cam_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:,0]])
                global_idx_str = ', '.join([str(int(tensor)) for tensor in frame_info[:,1]])
                x_label = "cams: {}".format(cam_idx_str)
            else:
                x_label = ""
            tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'], x_label=x_label)

        if key in ['img']:
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Input {}".format(key), size=title_font_size, y=0.79)
            grid_gray = grid_t  # torch.mean(grid_t, dim=0, keepdim=True)*0.333+grid_t*0.333+0.334
            tensor_imshow_normalized(ax_img, grid_gray, mean=config_dict['img_mean'],
                                                   stdDev=config_dict['img_std'], x_label=x_label, clip=True)
            height, width = images_fg.shape[2:4]
            plotTransformerBatch(ax_img, outputs_dict['spatial_transformer'].cpu().data, width, height)

    ############### labels_raw ################
    # plot 3D pose labels_raw
    for key in labels_raw.keys():
        if key in ['3D']:
            lable_pose = labels_raw[key]

            if lable_pose is not None:
                created_sub_plots += 1
                ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
                ax_3d_l.set_title("3D pose labels_raw", size=title_font_size)
                plot_3Dpose_batch(ax_3d_l, lable_pose.data.cpu().numpy(), bones=config_dict['bones'], radius=0.01, colormap='hsv')
                ax_3d_l.invert_zaxis()
                ax_3d_l.grid(False)
                if 1: # display a rotated version
                    created_sub_plots += 1
                    ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
                    ax_3d_l.set_title("3D pose labels_raw (rotated)", size=title_font_size)
                    a = -np.pi/2
                    R = np.array([[np.cos(a),0,-np.sin(a)],
                                  [0,1,0],
                                  [np.sin(a),0, np.cos(a)]])
                    pose_orig = lable_pose.data.cpu().numpy()
                    pose_rotated = pose_orig.reshape(-1,3) @ R.T
                    plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(pose_orig.shape), bones=config_dict['bones'], radius=0.01, colormap='hsv')
                    ax_3d_l.invert_zaxis()
                    ax_3d_l.grid(False)

    ############### network output ################
    # 3D pose label
    #train_crop_relative = hasattr(self, 'train_crop_relative') and self.train_crop_relative
    if '3D' in outputs_dict.keys():
        outputs_pose = outputs_dict['3D']
        outputs_pose = outputs_pose.cpu().data
        if config_dict['train_scale_normalized'] == 'mean_std':
            outputs_pose = denormalize_mean_std_tensor(outputs_pose, labels_raw)

        created_sub_plots += 1
        ax_3dp_p   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
        ax_3dp_p.set_title("3D prediction", size=title_font_size)
        plot_3Dpose_batch(ax_3dp_p, outputs_pose.numpy(), bones=config_dict['bones'], radius=0.01, colormap='hsv')
        ax_3dp_p.invert_zaxis()
        ax_3dp_p.grid(False)
        if 1: # display a rotated version
            created_sub_plots += 1
            ax_3d_l   = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots, projection='3d')
            ax_3d_l.set_title("3D pose prediction (rotated)", size=title_font_size)
            a = -np.pi/2
            R = np.array([[np.cos(a),0,-np.sin(a)],
                          [0,1,0],
                          [np.sin(a),0, np.cos(a)]])
            pose_rotated = outputs_pose.numpy().reshape(-1,3) @ R.T
            plot_3Dpose_batch(ax_3d_l, pose_rotated.reshape(outputs_pose.numpy().shape), bones=config_dict['bones'], radius=0.01, colormap='hsv')
            ax_3d_l.invert_zaxis()
            ax_3d_l.grid(False)

    # generated image
    for key in ['img','img_crop','img_downscaled','bg_crop','bg','blend_mask','blend_mask_crop','depth_map','spatial_transformer_img_crop','smooth_mask']:
        if key in outputs_dict.keys():
            images_out  = outputs_dict[key].cpu().data
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows,num_subplots_columns,created_sub_plots)
            ax_img.set_title("Output "+key, size=title_font_size)
            grid_t = torchvision.utils.make_grid(images_out, padding=0)
            if key in ['smooth_mask']: # only a single image in this case, constant
                ax_img.imshow(images_out)
                continue
            elif key in ['blend_mask_crop','blend_mask','smooth_mask']: # don't denormalize in this case
                tensor_imshow(ax_img, grid_t)
            else:
                tensor_imshow_normalized(ax_img, grid_t, mean=config_dict['img_mean'], stdDev=config_dict['img_std'])

    key = 'ST_depth'
    if key in outputs_dict.keys():
        created_sub_plots += 1
        ST_depth = outputs_dict[key].cpu().data
        ST_size = ST_depth.shape[0]
        width = 0.5 / ST_size
        ax = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        batch_size = ST_depth.size()[1]
        row_break = batch_size  # disables break, was not working with box plot...
        num_rows = batch_size // row_break
        ind = np.arange(batch_size)
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'purple']
        # for row in range(num_rows):
        for STi in range(ST_size):
            # left = row*row_break
            # right = (row+1)*row_break
            # rects = ax.bar(ind + STi*width - width*ST_size/2, 1*row+ST_depth[STi,left:right], width,
            #            color=colors[STi], label='ST_{}'.format(STi))
            rects = ax.bar(ind + STi * width - width * ST_size / 2, ST_depth[STi, :], width,
                           color=colors[STi], label='ST_{}'.format(STi))
        #        start, end = ax.get_xlim()
        #        ax.xaxis.set_ticks(np.arange(start, end, 1))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(1)
        ax.set_title("Output {}".format(key), size=title_font_size, y=0.79)

    key = 'similarity_matrix'
    if key in outputs_dict.keys():
        similarity_matrix = outputs_dict[key].cpu().data.permute(2, 0, 1).unsqueeze(1)
        grid_t = torchvision.utils.make_grid(similarity_matrix, padding=1, pad_value=0)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("Output {}".format(key), size=title_font_size, y=0.79)
        tensor_imshow(ax_img, grid_t)

    key = 'radiance_normalized'
    if key in outputs_dict.keys():
        img_shape = outputs_dict[key].shape[-2:]
        transmittance_normalized = outputs_dict[key].cpu().data.transpose(1, 0).contiguous().view(-1, 1,
                                                                                                  img_shape[0],
                                                                                                  img_shape[1])
        grid_t = torchvision.utils.make_grid(transmittance_normalized, padding=1, pad_value=0)
        created_sub_plots += 1
        ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
        ax_img.set_title("Output {}".format(key), size=title_font_size, y=0.79)
        tensor_imshow(ax_img, grid_t)

    shuffle_keys = ['shuffled_pose', 'shuffled_pose_inv', 'shuffled_appearance']
    for key in shuffle_keys:
        if key in outputs_dict.keys():
            indices = outputs_dict[key]
            permutation_matrix = np.zeros([len(indices), len(indices)])
            for i, j in enumerate(indices):
                permutation_matrix[i, j] = 1
            created_sub_plots += 1
            ax_img = fig.add_subplot(num_subplots_rows, num_subplots_columns, created_sub_plots)
            ax_img.set_title("Output {}".format(key), size=title_font_size, y=0.79)
            ax_img.imshow(permutation_matrix)

    if plot_iol.created_sub_plots_last[keyword] == created_sub_plots: # Don't save the dummy run that determines the number of plots
        plt.savefig(image_name,  dpi=config_dict['dpi'], transparent=True)
        print("Written image to {} at dpi={}".format(os.path.abspath(image_name), config_dict['dpi']))

    if verbose:
        plt.show()
    plt.close("all")
    plot_iol.created_sub_plots_last[keyword] = created_sub_plots
