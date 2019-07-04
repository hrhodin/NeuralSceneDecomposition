import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
torch.cuda.current_device() # to prevent  "Cannot re-initialize CUDA in forked subprocess." error on some configurations

import numpy as np
import numpy.linalg as la
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import plotting as utils_plt
from utils import skeleton as utils_skel

import train_detect_encode_decode
from ignite._utils import convert_tensor
from ignite.engine import Events

from matplotlib.widgets import Slider, Button

# load data
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTestNVS(train_detect_encode_decode.IgniteTrainNVS):
    def run(self, config_dict_file, config_dict):
        if 1: # load small example data
            import pickle
            data = pickle.load(open('../examples/test_set.pickl',"rb"))
            data = [{k:torch.FloatTensor(v) for k,v in d.items()} for d in data] # numpy to torch
        else:
            data_loader = self.load_data_test(config_dict)
            # save example data
            if 0:
                import pickle
                IPython.embed()
                data_iterator = iter(data_loader)
                data_cach = [next(data_iterator) for i in range(3)]
                data_cach = tuple(data_cach)
                pickle.dump(data_cach, open('../examples/test_set.pickl', "wb"))

        # load model
        model = self.load_network(config_dict)
        model = model.to(device)

        def tensor_to_npimg(torch_array):
            return np.swapaxes(np.swapaxes(torch_array.numpy(), 0, 2), 0, 1)

        def denormalize(np_array):
            return np_array * np.array(config_dict['img_std']) + np.array(config_dict['img_mean'])

        # extract image
        def tensor_to_img(output_tensor):
            output_img = tensor_to_npimg(output_tensor)
            output_img = denormalize(output_img)
            output_img = np.clip(output_img, 0, 1)
            return output_img

        def rotationMatrixXZY(theta, phi, psi):
            Ax = np.matrix([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
            Ay = np.matrix([[np.cos(phi), 0, -np.sin(phi)],
                            [0, 1, 0],
                            [np.sin(phi), 0, np.cos(phi)]])
            Az = np.matrix([[np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1], ])
            return Az * Ay * Ax

        # get next image
        input_dict, label_dict = None, None
        image_index = -1
        input_dict, label_dict = data
        def nextImage():
            nonlocal image_index
            image_index += 1
            #nonlocal input_dict, label_dict
            #input_dict, label_dict = next(data_iterator)
            input_dict['external_rotation_global'] = torch.from_numpy(np.eye(3)).float().to(device)
        nextImage()


        # apply model on images
        output_dict = None
        def predict():
            nonlocal output_dict
            model.eval()
            with torch.no_grad():
                input_dict_cuda, label_dict_cuda = utils_data.nestedDictToDevice((input_dict, label_dict), device=device)
                output_dict_cuda = model(input_dict_cuda)
                output_dict = utils_data.nestedDictToDevice(output_dict_cuda, device='cpu')
        predict()

        # init figure
        my_dpi = 400
        fig, ax_blank = plt.subplots(figsize=(5 * 800 / my_dpi, 5 * 300 / my_dpi))
        plt.axis('off')
        title_font_size = 5

        # input image
        ax_in_img = plt.axes([-0.05, 0.55, 0.4, 0.4])
        ax_in_img.axis('off')
        im_input = plt.imshow(tensor_to_img(input_dict['img'][image_index]), animated=True)
        ax_in_img.set_title("Input img", size=title_font_size)

        # input image with bounding box
        ax_in_img2 = plt.axes([0.25, 0.55, 0.4, 0.4])
        ax_in_img2.axis('off')
        im_input2 = plt.imshow(tensor_to_img(input_dict['img'][image_index]), animated=True)
        ax_in_img2.set_title("Bounding box", size=title_font_size)
        height, width = input_dict['img'][image_index].shape[1:3]
        rect = self.plotTransformer(ax_in_img2, image_index, output_dict['spatial_transformer'], width, height)

        # output image
        key = 'img'
        if key in output_dict:
            ax_out_img = plt.axes([0.55, 0.55, 0.4, 0.4])
            ax_out_img.axis('off')
            img_out = tensor_to_img(output_dict[key][image_index])
            handle_im_out = plt.imshow(img_out, animated=True)
            ax_out_img.set_title("Output img", size=title_font_size)

            # output seg
            ax_out_seg = plt.axes([0.25, 0.08, 0.4, 0.4])
            ax_out_seg.axis('off')
            color = output_dict['blend_mask'][0].cpu().data.permute(1, 2, 0)
            handle_seg_out = plt.imshow(color, animated=True)
            ax_out_seg.set_title("Intance segmentation", size=title_font_size)

            # output depth
            ax_out_depth = plt.axes([0.55, 0.08, 0.4, 0.4])
            ax_out_depth.axis('off')
            img_out = tensor_to_img(output_dict['depth_map'][image_index])
            handle_depth_out = plt.imshow(img_out, animated=True)
            ax_out_depth.set_title("Depth map", size=title_font_size)

        key = '3D'
        if key in output_dict:
            # output skeleton 1
            ax_pred_skel1 = fig.add_subplot(111, projection='3d')
            ax_pred_skel1.set_position([0.6, 0.5, 0.1, 0.4])
            handle_pred_skel1 = utils_plt.plot_3Dpose_simple(ax_pred_skel1, label_dict['3D'][image_index].numpy().reshape([-1, 3]).T,
                                                           bones=utils_skel.bones_h36m, linewidth=5,
                                                           plot_handles=None)  # , colormap='Greys')
            ax_pred_skel1.invert_zaxis()
            ax_pred_skel1.grid(False)
            ax_pred_skel1.set_axis_off()
            ax_pred_skel1.set_title("Pred. (left)", size=title_font_size)

            # output skeleton 2
            ax_pred_skel2 = fig.add_subplot(111, projection='3d')
            ax_pred_skel2.set_position([0.7, 0.5, 0.1, 0.4])
            handle_pred_skel2 = utils_plt.plot_3Dpose_simple(ax_pred_skel2, label_dict['3D'][image_index].numpy().reshape([-1, 3]).T,
                                                           bones=utils_skel.bones_h36m, linewidth=5,
                                                           plot_handles=None)  # , colormap='Greys')
            ax_pred_skel2.invert_zaxis()
            ax_pred_skel2.grid(False)
            ax_pred_skel2.set_axis_off()
            ax_pred_skel2.set_title("(right)", size=title_font_size)

            # gt skeleton 1
            ax_gt_skel2 = fig.add_subplot(111, projection='3d')
            ax_gt_skel2.set_position([0.8, 0.5, 0.1, 0.4])
            handle_gt_skel2 = utils_plt.plot_3Dpose_simple(ax_gt_skel2, label_dict['3D'][image_index].numpy().reshape([-1, 3]).T,
                                                           bones=utils_skel.bones_h36m, linewidth=5,
                                                           plot_handles=None)  # , colormap='Greys')
            ax_gt_skel2.invert_zaxis()
            ax_gt_skel2.grid(False)
            ax_gt_skel2.set_axis_off()
            ax_gt_skel2.set_title("GT (left)", size=title_font_size)

            # gt skeleton 2
            ax_gt_skel1 = fig.add_subplot(111, projection='3d')
            ax_gt_skel1.set_position([0.9, 0.5, 0.1, 0.4])
            handle_gt_skel1 = utils_plt.plot_3Dpose_simple(ax_gt_skel1, label_dict['3D'][image_index].numpy().reshape([-1, 3]).T,
                                                           bones=utils_skel.bones_h36m, linewidth=5,
                                                           plot_handles=None)  # , colormap='Greys')
            ax_gt_skel1.invert_zaxis()
            ax_gt_skel1.grid(False)
            ax_gt_skel1.set_axis_off()
            ax_gt_skel1.set_title("(right)", size=title_font_size)

        # update figure with new data
        def update_figure():
            # images
            im_input.set_array(tensor_to_img(input_dict['img'][image_index]))
            im_input2.set_array(tensor_to_img(input_dict['img'][image_index]))
            self.plotTransformer(ax_in_img2, image_index, output_dict['spatial_transformer'], width, height, rect)

            key = 'img'
            if key in output_dict:
                handle_im_out.set_array(tensor_to_img(output_dict[key][image_index]))
                handle_seg_out.set_array(output_dict['blend_mask'][image_index].cpu().data.permute(1, 2, 0))
                handle_depth_out.set_array(tensor_to_img(output_dict['depth_map'][image_index]))

            key = '3D'
            if key in output_dict:
                # TODO: why ordered right to left here?
                # gt 3D poses
                gt_pose = label_dict[key][image_index]
                #IPython.embed()
                R_cam_2_world = input_dict['R_cam_2_world'][image_index].numpy()
                R_world_in_cam = la.inv(R_cam_2_world) @ input_dict['external_rotation_global'].cpu().numpy() @ R_cam_2_world
                pose_rotated1 = R_world_in_cam @ gt_pose[1].numpy().reshape([-1, 3]).T
                utils_plt.plot_3Dpose_simple(ax_gt_skel1, pose_rotated1, bones=utils_skel.bones_h36m,
                                             plot_handles=handle_gt_skel1)
                pose_rotated2 = R_world_in_cam @ gt_pose[0].numpy().reshape([-1, 3]).T
                utils_plt.plot_3Dpose_simple(ax_gt_skel2, pose_rotated2, bones=utils_skel.bones_h36m,
                                             plot_handles=handle_gt_skel2)

                # prediction 3D poses
                pose_mean = label_dict['pose_mean'][image_index].numpy()
                pose_std = label_dict['pose_std'][image_index].numpy()
                pred_pose1 = (output_dict[key][image_index].numpy().reshape(pose_mean.shape) * pose_std) + pose_mean
                pose_rotated1 = R_world_in_cam @ pred_pose1[0].reshape([-1, 3]).T
                utils_plt.plot_3Dpose_simple(ax_pred_skel1, pose_rotated1, bones=utils_skel.bones_h36m,
                                             plot_handles=handle_pred_skel1)
                pred_pose2 = (output_dict[key][image_index].numpy().reshape(pose_mean.shape) * pose_std) + pose_mean
                pose_rotated2 = R_world_in_cam @ pred_pose2[1].reshape([-1, 3]).T
                utils_plt.plot_3Dpose_simple(ax_pred_skel2, pose_rotated2, bones=utils_skel.bones_h36m,
                                             plot_handles=handle_pred_skel2)

            # flush drawings
            fig.canvas.draw_idle()
            print("using",device,"device")

        update_figure()

        def update_rotation(event):
            rot = slider_yaw_glob.val
            print("Rotationg ",rot)
            batch_size = input_dict['img'].size()[0]
            input_dict['external_rotation_global'] = torch.from_numpy(rotationMatrixXZY(theta=0, phi=rot, psi=0)).float().to(device) # boxing coords
            #input_dict['external_rotation_global'] = torch.from_numpy(rotationMatrixXZY(theta=0, phi=0, psi=rot)).float().to(device) # H36m
            input_dict['external_rotation_cam'] = torch.from_numpy(np.eye(3)).float().to(device) # torch.from_numpy(rotationMatrixXZY(theta=0, phi=rot, psi=0)).float().cuda()
            predict()
            update_figure()

        ax_next = plt.axes([0.05, 0.03, 0.15, 0.04])
        button_next = Button(ax_next, 'Next image', color='lightgray', hovercolor='0.975')
        def nextButtonPressed(event):
            nextImage()
            predict()
            update_figure()
        button_next.on_clicked(nextButtonPressed)
        ax_yaw_glob = plt.axes([0.25, 0.03, 0.65, 0.015], facecolor='lightgray')
        slider_range = np.pi
        slider_yaw_glob = Slider(ax_yaw_glob, 'Yaw', -slider_range, slider_range, valinit=0)
        slider_yaw_glob.on_changed(update_rotation)
        plt.show()

    def plotTransformer(self, ax_img, i, transformation, width, height, rects=None):
        num_transformers = transformation.size()[0]
        clist = ['red','green','blue','orange','cyan','magenta','black','white']
        linewidth = 2
        if rects is None:
            rects = []
        for j in range(num_transformers):
            affine_matrix = transformation[j, i]
            x_scale = affine_matrix[0, 0].item()
            y_scale = affine_matrix[1, 1].item()
            x_relative = affine_matrix[0, 2].item()
            y_relative = affine_matrix[1, 2].item()
            color = clist[j]
            if j>=len(rects):
                rects.append(patches.Rectangle( (0,0),1,1, linewidth=linewidth, linestyle='dashed', edgecolor=color, facecolor='none'))
                ax_img.add_patch(rects[j])
            rects[j].set_bounds(
                width  * ((x_relative + 1 - x_scale) / 2),
                 height * ((y_relative + 1 - y_scale) / 2),
                x_scale * width, y_scale * height)
        return rects


if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_test_detect_encode_decode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTestNVS()
    ignite.run(config_dict_module.__file__, config_dict)
