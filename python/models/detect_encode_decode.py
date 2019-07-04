import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Dropout

import IPython
import torch
import torch.nn.functional as F
import numpy as np

from models import resnet_transfer
from models import resnet_VNECT_3Donly

import models.unet_utils as utils_unet

from models import MLP

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


class decoder_unet(nn.Module):
    def __init__(self, bottlneck_feature_dim, num_encoding_layers, output_channels, is_deconv):
        super(decoder_unet, self).__init__()
        self.num_encoding_layers = num_encoding_layers
        self.is_deconv = is_deconv

        self.filters = [64, 128, 256, 512, 512, 512]
        self.feature_scale = self.filters[-1] // bottlneck_feature_dim
        assert self.feature_scale == self.filters[-1] / bottlneck_feature_dim # integer division?
        self.filters = [x // self.feature_scale for x in self.filters]

        upper_conv = self.is_deconv and not upper_billinear
        lower_conv = self.is_deconv and not lower_billinear

        for li in range(1, num_encoding_layers - 1):
            setattr(self, 'upconv_' + str(li) + '_stage',
                    utils_unet.unetUpNoSKip(self.filters[num_encoding_layers - li], self.filters[num_encoding_layers - li - 1],
                                 upper_conv, padding=1, batch_norm=False))

        setattr(self, 'upconv_' + str(num_encoding_layers - 1) + '_stage',
                utils_unet.unetUpNoSKip(self.filters[1], self.filters[0], lower_conv, padding=1, batch_norm=False))

        setattr(self, 'final_stage', nn.Conv2d(self.filters[0], output_channels, 1))

        self.relu  = ReLU(inplace=True)
        self.relu2 = ReLU(inplace=False)
        self.dropout = Dropout(inplace=True, p=0.3)

        self.reset_params() # HACK

    def reset_params(self):
        # note, self.modules() lists all moduls and their submodules, hence no recursion is necessary
        for i, m in enumerate(self.modules()):
            utils_unet.weight_init(m)

    def forward(self, x):
        out_deconv = x

        for li in range(1, self.num_encoding_layers - 1):
            out_deconv = getattr(self, 'upconv_' + str(li) + '_stage')(out_deconv)

        out_deconv = getattr(self, 'upconv_' + str(self.num_encoding_layers - 1) + '_stage')(
            out_deconv)

        return getattr(self, 'final_stage')(out_deconv)


def construct_bump_function(in_resolution, type):
    if type == True:
        return None
    xs = torch.linspace(-1, 1, in_resolution, dtype=torch.float)
    xs = xs.view(1, -1).expand(in_resolution, in_resolution)
    ys = torch.linspace(-1, 1, in_resolution, dtype=torch.float)
    ys = ys.view(-1, 1).expand(in_resolution, in_resolution)
    if type == 'Gauss':
        bump_function = torch.exp(- 2 * (xs ** 2 + ys ** 2))  # Gaussian with std=0.5
    elif type == 'GaussB':
        bump_function = torch.exp(
            1 - 1 / (1 - (xs ** 2 + ys ** 2)))  # classical bump function, rescaled to have maxima == 1
    elif type == 'GaussBsq':
        bump_function = torch.exp(1 - 1 / (1 - (xs ** 4 + ys ** 4)))  # sharper version with larger plateau
    elif type == 'GaussBSqSqr':
        bump_function = torch.exp(1 - 1 / (1 - np.sqrt(xs ** 4 + ys ** 4)))  # sharper version with larger plateau
    else:
        bump_function = torch.ones(xs.shape)

    # bump_function = torch.exp(1 - 1 / (1 - np.sqrt(xs**4 + ys**4))) # going more into the corner than the classical one
    # handle limit of x->+-1 and y->+-1, where 1/(1-1) leads to nans
    for r in range(in_resolution):
        for c in range(in_resolution):
            if type == 'Gauss':
                continue
            if type == 'GaussB':
                put_zero = (xs[r, c] ** 2 + ys[r, c] ** 2) >= 1
            if type == 'GaussBsq':
                put_zero = (xs[r, c] ** 4 + ys[r, c] ** 4) >= 1
            if type == 'GaussBSqSqr':
                put_zero = np.sqrt(xs[r, c] ** 4 + ys[r, c] ** 4) >= 1
            if put_zero:
                bump_function[r, c] = 0
    return bump_function.to(device)

# reshapes the tensor by merging the specified dim with the following one
def mergeDim(tensor,dim=0):
    s = tensor.shape
    s_new = [*s[:dim],s[dim]*s[dim+1]]
    if dim+2<len(s):
        s_new+=s[dim+2:]
    return tensor.view(s_new)

# transposes two dimensions, flattens these in the mask and data tensor, and selects the masked indices
def transposed_mask_select(x,mask,tdims):
    x_prep    = mergeDim(   x.transpose(*tdims).contiguous(), dim=tdims[0])
    mask_prep = mergeDim(mask.transpose(*tdims).contiguous(), dim=tdims[0])
    x_new = torch.masked_select(x_prep,mask=mask_prep)
    return x_new

class detect_encode_decode(nn.Module):
    bump_function = None # static variable
    def __init__(self, feature_scale=4,  # to reduce dimensionality
                 in_resolution=256,
                 output_channels=3, is_deconv=True,
                 upper_billinear=False,
                 lower_billinear=False,
                 in_channels=3, is_batchnorm=True,
                 num_joints=17, nb_dims=3,  # ecoding transformation
                 num_encoding_layers=5,
                 dimension_bg=256,
                 dimension_fg=256,
                 dimension_3d=3 * 64,  # needs to be devidable by 3
                 latent_dropout=0.3,
                 shuffle_fg=True,
                 shuffle_3d=True,
                 n_hidden_to3Dpose=2,
                 subbatch_size=4,
                 implicit_rotation=False,
                 spatial_transformer=False,
                 ST_size=1,
                 spatial_transformer_bounds=1,
                 masked_blending=True,
                 scale_mask_max_to_1=True,
                 output_types=['3D', 'img', 'shuffled_pose', 'shuffled_appearance'],
                 predict_transformer_depth=False,
                 normalize_mask_density=False,
                 match_crops=False,
                 offset_crop=False,
                 mode='NVS',
                 transductive_training=[],
                 similarity_bandwidth=10,
                 disable_detector=False,
                 ):
        super(detect_encode_decode, self).__init__()
        arguments = []
        print("settign locals")
        for k,v in list(locals().items()):
            if k in "self":
                continue
            setattr(self, k, v)

        assert dimension_3d % 3 == 0
        self.match_crops = match_crops and ST_size>1

        self.bottlneck_feature_dim = 512 // feature_scale
        self.bottleneck_resolution = in_resolution // (2 ** (num_encoding_layers - 1))
        num_bottlneck_features = self.bottleneck_resolution**2 * self.bottlneck_feature_dim
        print('bottleneck_resolution', self.bottleneck_resolution, 'num_bottlneck_features', num_bottlneck_features)

        ################################################
        ############ Spatial transformer ###############
        if self.spatial_transformer:
            params_per_transformer = 4# translation x, translation y, and scale x and y
            #if self.predict_transformer_depth: Note, commented to always always predict depth, to allow weight transfer
            params_per_transformer += 1
            affine_dimension = ST_size*params_per_transformer
            self.detection_resolution = self.in_resolution
            self.detector = resnet_transfer.resnet18(num_classes=affine_dimension, num_channels=3,
                                                     input_width=self.detection_resolution, nois_stddev=0,
                                                     output_key='affine')

            # init spatial transformer to be centered and relatively small
            self.detector.fc.weight.data.zero_()
            s = 0.5
            self.detector.fc.bias.data.zero_()
            self.detector.fc.bias.data.view([ST_size, params_per_transformer])[:,0] = s # ones from affine matrix flattened: [s, 0, 0, 0, s, 0]
            self.detector.fc.bias.data.view([ST_size, params_per_transformer])[:,1] = s

            # construct the bump function
            detect_encode_decode.bump_function = construct_bump_function(self.in_resolution, type=self.spatial_transformer)

        ####################################
        ############ encoder ###############
        self.encoder = resnet_VNECT_3Donly.resnet50(pretrained=True, input_key='img_crop',
                                                   output_keys=['latent_3d', '2D_heat'],
                                                   input_width=in_resolution, #net_type='high_res',
                                                   num_classes=self.dimension_fg + self.dimension_3d)

        ##################################################
        ############ latent transformation ###############
        assert self.dimension_fg < self.bottlneck_feature_dim
        num_bottlneck_features_3d = self.bottleneck_resolution**2 * (self.bottlneck_feature_dim - self.dimension_fg)

        self.to_3d = nn.Sequential(Linear(num_bottlneck_features, self.dimension_3d),
                                   Dropout(inplace=True, p=self.latent_dropout)  # removing dropout degrades results
                                   )

        if self.dimension_fg > 0:
            self.to_fg = nn.Sequential(Linear(num_bottlneck_features, self.dimension_fg),
                                       Dropout(inplace=True, p=self.latent_dropout),
                                       ReLU(inplace=False))
        self.from_latent = nn.Sequential(Linear(self.dimension_3d, num_bottlneck_features_3d),
                                         Dropout(inplace=True, p=self.latent_dropout),
                                         ReLU(inplace=False))

        ####################################
        ############ decoder ###############
        if self.mode == 'pose':
            self.pose_decoder = MLP.MLP_fromLatent(d_in=self.dimension_3d, d_hidden=2048, d_out=51,
                                                   n_hidden=n_hidden_to3Dpose,
                                                   dropout=0.5)
        else:
            if self.masked_blending:
                output_channels_combined = output_channels + 1
            else:
                output_channels_combined = output_channels
            self.decoder = decoder_unet(bottlneck_feature_dim=self.bottlneck_feature_dim,
                                        num_encoding_layers=num_encoding_layers,
                                        output_channels=output_channels_combined,
                                        is_deconv=is_deconv)

    def roll_segment_random(self, list, start, end):
        selected = list[start:end]
        if self.training:
            if np.random.random([1])[0] < 0.5: # 50% rotation worked well, by percentage of camera breaks..
                selected = np.roll(selected, 1).tolist() # flip (in case of pairs)
        else:  # deterministic shuffling for testing
            selected = np.roll(selected, 1).tolist()
        list[start:end] = selected

    def flip_segment(self, list, start, width):
        selected = list[start:start + width]
        list[start:start + width] = list[start + width:start + 2 * width]
        list[start + width:start + 2 * width] = selected


    def forward(self, input_dict):
        if self.mode == 'pose':
            return self.forward_pose(input_dict)
        else:
            return self.forward_NVS(input_dict)

    def forward_detector(self, input_dict, shuffled_crops=None):
        # downscale input image before running the detector
        input_img = input_dict['img'].squeeze()
        batch_size = input_dict['img'].shape[0]
        ST_size = self.ST_size
        def ST_flatten(batch):
            shape_orig = batch.shape
            #assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size*batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        ### downscale image ###
        affine_matrix_downscale = torch.FloatTensor([[1, 0, 0], [0, 1, 0]])
        affine_matrix_downscale = affine_matrix_downscale.unsqueeze(0).repeat(batch_size, 1, 1).to(device)

        # simulate randomized crops to regularize training
        if self.offset_crop and self.training:
            offsets = torch.FloatTensor((np.random.random([batch_size,2])-0.5)*0.1).to(device)
            affine_matrix_downscale[:,:2,2] = offsets

        size = torch.Size([batch_size, 3, self.detection_resolution, self.detection_resolution])
        grid_downscale = F.affine_grid(affine_matrix_downscale, size=size)
        input_img_downscaled = F.grid_sample(input_img, grid_downscale, padding_mode='zeros')

        ### run detector ###
        affine_params = self.detector.forward(input_img_downscaled)['affine']
        affine_dim_total = affine_params.shape[1]
        # separate params for multiple detections
        affine_params = affine_params.view(batch_size * ST_size, affine_dim_total // ST_size)

        # apply non-linearities and extract depth
        border_factor = self.spatial_transformer_bounds['border_factor']
        min_size = self.spatial_transformer_bounds['min_size']
        max_size = self.spatial_transformer_bounds['max_size']
        scales = min_size + (max_size-min_size) * F.sigmoid(
            affine_params[:, 0:2])  # possible range 0..1, further restricted to be not too small/big
        #max_positions = 1.0 - 0.0 * scales  # -1.1scale worked well, -scale to offset from the boundary, -0.9*scale meanse at most 10% out of view, 1.1scale means 10% away from the border
        # Note, c*scale with c<1 creates a bias to get stuck at a corner, since then the overlay image is smaller. This would need to be peanalized
        max_positions = border_factor
        position = max_positions * F.tanh(
            affine_params[:, 2:4])  # possible range -1..1, further reduced by max_position to keep partially in view

        # undo crop jitter of input frame
        if self.offset_crop and self.training:
            for STi in range(ST_size):
                position.view([batch_size, ST_size, -1])[:,STi,:] += offsets

        affine_matrix_crop = torch.zeros([batch_size * ST_size, 2, 3]).to(device)
        affine_matrix_crop[:, 0, 0] = scales[:, 0]
        affine_matrix_crop[:, 1, 1] = scales[:, 1]
        affine_matrix_crop[:, 0:2, 2] = position[:, :2]  # normalized to -1..1
        if self.predict_transformer_depth:
            ST_depth = affine_params[:, 4]
            ST_depth = ST_depth.view(batch_size, ST_size).transpose(1, 0).contiguous()
        else:
            ST_depth = None

        if self.disable_detector:
            affine_matrix_crop[:, 0, 0] = 1
            affine_matrix_crop[:, 1, 1] = 1
            affine_matrix_crop[:, 0:2, 2] = 0  # normalized to -1..1
        affine_matrix_crop_raw = affine_matrix_crop

        # use the GT
        if 'spatial_transformer' in input_dict:
            affine_matrix_crop = input_dict['spatial_transformer']

        # inverse of the affine transformation (exploiting the near-diagonal structure)
        affine_matrix_uncrop = torch.zeros(affine_matrix_crop.shape).to(device)
        affine_matrix_uncrop[:, 0, 0] = 1 / affine_matrix_crop[:, 0, 0]
        affine_matrix_uncrop[:, 1, 1] = 1 / affine_matrix_crop[:, 1, 1]
        affine_matrix_uncrop[:, 0, 2] = -affine_matrix_crop[:, 0, 2] / affine_matrix_crop[:, 0, 0]
        affine_matrix_uncrop[:, 1, 2] = -affine_matrix_crop[:, 1, 2] / affine_matrix_crop[:, 1, 1]

        if 0 and 'shuffled_appearance_weight' in input_dict.keys(): # HACK for view interpolation
            w = input_dict['shuffled_appearance_weight'].item()
            a0 = (1 - w) * affine_matrix_uncrop[0, :, :] + w * affine_matrix_uncrop[2, :, :] # blend between neighboring frames, same ST
            a1 = (1 - w) * affine_matrix_uncrop[1, :, :] + w * affine_matrix_uncrop[3, :, :] # blend between neighboring frames, same ST
            affine_matrix_uncrop[0, :, :] = a0
            affine_matrix_uncrop[1, :, :] = a1

        # make the ST dimension the first one, as needed for cropping
        affine_matrix_crop_multi = affine_matrix_crop.view(batch_size, ST_size, 2, 3).transpose(1, 0).contiguous()
        affine_matrix_uncrop_multi = affine_matrix_uncrop.view(batch_size, ST_size, 2, 3).transpose(1, 0).contiguous()

        # apply spatial transformers (crop input and bg images)
        img_crop = []
        bg_crop  = []
        R_virt2orig_list = []
        P_virt2orig_list = []
        for j in range(ST_size):
            output_size = torch.Size([batch_size, 3, self.in_resolution, self.in_resolution])
            grid_crop = F.affine_grid(affine_matrix_crop_multi[j, :, :, :], size=output_size)

            img_crop.append(F.grid_sample(input_img, grid_crop))
            if 'bg' in input_dict:
                bg_img = input_dict['bg']
                bg_crop.append(F.grid_sample(bg_img, grid_crop))

        # concatenate spatial transformers along batch size
        if 'bg' in input_dict:
            bg_crop = torch.cat(bg_crop)
        img_crop = torch.cat(img_crop)

        # apply smooth spatial transformer in forward pass
        if self.spatial_transformer and detect_encode_decode.bump_function is not None:
            img_crop = img_crop * detect_encode_decode.bump_function.unsqueeze(0).unsqueeze(0)

        if 0: # check disabled, might be slow
            if torch.isnan(img_crop).any():
                print('WARNING: unet_encode3D_V2 forward_detector: torch.isnan(img_crop)')
                IPython.embed()
                eliminateNaNs(img_crop)

        input_dict_cropped = {'img_crop': img_crop,
                              'bg_crop': bg_crop}

        return input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw, grid_crop

    def forward_NVS(self, input_dict):
        if 'img' in input_dict.keys():
            batch_size = input_dict['img'].size()[0]
        else:
            batch_size = input_dict['img_crop'].size()[0]
        ST_size = self.ST_size

        num_pose_examples = batch_size // 2
        num_appearance_examples = batch_size // 2
        num_appearance_subbatches = num_appearance_examples // np.maximum(self.subbatch_size, 1)

        def ST_flatten(batch):
            shape_orig = batch.shape
            #assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size*batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        def features_flatten(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1]
            return batch.view(shape_new)

        def features_split3D(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1,3]
            return batch.view(shape_new)

        ########################################################
        # Determine shuffling
        shuffled_appearance = list(range(batch_size))
        shuffled_pose = list(range(batch_size))
        shuffled_crops = list(range(batch_size*ST_size))
        num_pose_subbatches = batch_size // np.maximum(self.subbatch_size, 1)

        if len(self.transductive_training):
            subj = input_dict['subj']

        # only if no user input is provided
        rotation_by_user = self.training == False and 'external_rotation_cam' in input_dict.keys()
        if not rotation_by_user:
            if self.shuffle_fg and self.training == True:
                for i in range(0, num_pose_subbatches):
                    self.roll_segment_random(shuffled_appearance, i * self.subbatch_size, (i + 1) * self.subbatch_size)
                for i in range(0, num_pose_subbatches // 2):  # flip first with second subbatch
                    self.flip_segment(shuffled_appearance, i * 2 * self.subbatch_size, self.subbatch_size)
            if self.shuffle_3d:
                for i in range(0, num_pose_subbatches):
                    # don't rotate on test subjects, to mimick that we don't know the rotation
                    if len(self.transductive_training)==0 or int(subj[i * self.subbatch_size]) not in self.transductive_training:
                        self.roll_segment_random(shuffled_pose, i * self.subbatch_size, (i + 1) * self.subbatch_size)

        # infer inverse mapping
        shuffled_pose_inv = [-1] * batch_size
        for i, v in enumerate(shuffled_pose):
            shuffled_pose_inv[v] = i

        shuffled_appearance = torch.LongTensor(shuffled_appearance).to(device)
        shuffled_pose = torch.LongTensor(shuffled_pose).to(device)
        shuffled_pose_inv = torch.LongTensor(shuffled_pose_inv).to(device)
        shuffled_crops = torch.LongTensor(shuffled_crops).to(device)

        shuffled_crops_inv = shuffled_crops.clone()
        for i, v in enumerate(shuffled_crops):
            shuffled_crops_inv[v] = i

        if rotation_by_user:
            if 'shuffled_appearance' in input_dict.keys():
                shuffled_appearance = input_dict['shuffled_appearance'].long()

        ###############################################
        # determine shuffled rotation
        cam_2_world = input_dict['R_cam_2_world'].view((batch_size, 3, 3)).float()
        world_2_cam = cam_2_world.transpose(1, 2)
        if rotation_by_user:
            external_cam = input_dict['external_rotation_cam'].view(1, 3, 3).expand((batch_size, 3, 3))
            external_glob = input_dict['external_rotation_global'].view(1, 3, 3).expand((batch_size, 3, 3))
            cam2cam = torch.bmm(external_cam, torch.bmm(world_2_cam, torch.bmm(external_glob, cam_2_world)))
        else:
            world_2_cam_shuffled = torch.index_select(world_2_cam, dim=0, index=shuffled_pose)
            cam2cam = torch.bmm(world_2_cam_shuffled, cam_2_world)

        # copy one cam2cam for each spatial transformer
        cam2cam = cam2cam.repeat([ST_size,1,1])

        ###############################################
        # spatial transformer
        if self.spatial_transformer:
            input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw, grid_crop  = self.forward_detector(input_dict, shuffled_crops)
            input_img_downscaled = input_img_downscaled.data.cpu()
        else:
            input_dict_cropped = input_dict  # fallback to using crops from dataloader

        ###############################################
        # encoding stage
        output = self.encoder.forward(input_dict_cropped)['latent_3d']
        has_fg = hasattr(self, "to_fg")
        if has_fg:
            latent_fg = output[:, :self.dimension_fg] #.contiguous().clone() # TODO
            latent_fg = ST_split(latent_fg)
        latent_3d = output[:, self.dimension_fg:self.dimension_fg + self.dimension_3d]
        latent_3d = features_split3D(ST_split(latent_3d)) # transform it into a 3D latent space

        ###############################################
        # latent rotation (to shuffled view)
        cam2cam_transposed = cam2cam.transpose(1, 2)
        latent_3d_rotated = torch.bmm(ST_flatten(latent_3d), cam2cam_transposed)
        latent_3d_rotated = ST_split(latent_3d_rotated)

        # user input to flip pose
        if 'shuffled_pose_weight' in input_dict.keys():
            w = input_dict['shuffled_pose_weight']
            # weighted average with the last one
            latent_3d_rotated = (1 - w.expand_as(latent_3d)) * latent_3d \
                                + w.expand_as(latent_3d) * latent_3d_rotated[-1:].expand_as(latent_3d)

        # shuffle appearance based on flipping indices or user input
        if has_fg:
            # shuffle the appearance for all candidate crops
            #latent_fg_time_shuffled = torch.index_select(latent_fg, dim=0, index=shuffled_appearance_multiple)
            latent_fg_time_shuffled = torch.index_select(latent_fg, dim=1, index=shuffled_appearance)
            # blend background during view transition
            if 0 and 'shuffled_appearance_weight' in input_dict.keys(): # HACK disabled!!
                w = input_dict['shuffled_appearance_weight'].item()
                latent_fg_time_shuffled = (1 - w) * latent_fg + w * latent_fg_time_shuffled

        # compute similarity matrix
        if self.match_crops and has_fg:
            # TODO: similarities across time
            latent_fg_target = latent_fg # this is the bbox and appearance information to which we decode
            latent_fg_source = torch.index_select(latent_fg, dim=1, index=shuffled_pose_inv) # the one we encode from

            # expand along dim 0 and 1 respectively to compute covariance
            square_shape = [ST_size] + list(latent_fg_target.shape)
            # Note, selecting only the first 16 (out of usually 128) to make not the whole appearance space dependent
            num_matching_channels = 16
            #num_matching_channels = 128
            latent_fg_target_exp = latent_fg_target.unsqueeze(1).expand(square_shape)[:, :, :, :num_matching_channels]
            # TODO: is this expand needed? Broadcasting should work..
            latent_fg_source_exp = latent_fg_source.unsqueeze(0).expand(square_shape)[:, :, :, :num_matching_channels]
            eps = 0.0001
            dot_product = torch.sum(latent_fg_source_exp * latent_fg_target_exp, dim=-1)
            angle_matrix = dot_product / (
                                     eps + torch.norm(latent_fg_source_exp, dim=-1)
                                         * torch.norm(latent_fg_target_exp, dim=-1))  # cos angle
            # defined in a way, that it is 1 if the two encodings are identical and -1 if they are opposing
            # the rows of the resulting matrix assign a weighted average to the other (rotated) view, i.e. the first rows represent the weights for the first crop in the output image
            similarity_matrix = angle_matrix  # torch.sum(correlation_matrix, dim=3)
            if 0: # disabled, might be slow
                if torch.isnan(similarity_matrix).any():
                    print('WARNING: torch.isnan(similarity_matrix)')
                    IPython.embed()

            def softmax2D(c):
                bandwidth = self.similarity_bandwidth #  was 2, higher values will lead to a sharper max
                c_sm0 = F.softmax(c*bandwidth, dim=1)
                # c_sm01 = F.softmax(c_sm0, dim=1)
                # c_sm1 = F.softmax(c,     dim=1)
                # c_sm10 = F.softmax(c_sm1, dim=0)
                #                return c_sm01 #TODO: which one to pick??
                #                return c_sm10 # this one should ensure sum(weights) == 1
                return c_sm0  # ensures weigths 1 per crop, but not exclusive

            def hardmax2D(c):
                c = softmax2D(c) # first to the usual softmax computation with badwidth
                max, arg_max = torch.max(c, dim=1, keepdim=True)
                c_sm0 = torch.zeros(c.shape).to(device)
                for i in range(batch_size):
                    for s in range(ST_size):
                        c_sm0[s, arg_max[s, 0, i], i] = 1
                return c_sm0
                # I was to stupid to get the arg_max working, attempt below:
                #
                # arg_max_flatten = arg_max.view(-1)
                # binary_similarity = torch.zeros(c.shape).to(device)
                # stride = float(np.prod(c[0].shape))
                # binary_similarity.view(-1)[arg_max_flatten*stride] = 0.5 # ST_size
                # torch.gather(binary_similarity,dim=0,index=arg_max) = 1

            if self.training:
                similarity_matrix_normalized = softmax2D(similarity_matrix)
            else:
                similarity_matrix_normalized = hardmax2D(similarity_matrix)
        else:
            # fixed assignment at test time
            similarity_matrix_normalized = torch.zeros([ST_size,ST_size,batch_size]).to(device)
            similarity_matrix_normalized[:, :, :] = 0
            for STi in range(ST_size):
                similarity_matrix_normalized[STi, STi, :] = 1


        ###############################################
        # decoding
        latent_combined = features_flatten(latent_3d_rotated)
        # Note, the unflattened version seems to work as well
        map_from_3d = ST_split(self.from_latent(ST_flatten(latent_combined)))
        map_width = self.bottleneck_resolution
        map_channels = self.bottlneck_feature_dim
        if has_fg:
            latent_fg_time_shuffled_replicated_spatially = latent_fg_time_shuffled.unsqueeze(-1).unsqueeze(-1).expand(
                ST_size, batch_size, self.dimension_fg, map_width, map_width)
            latent_shuffled = torch.cat([latent_fg_time_shuffled_replicated_spatially,
                                         map_from_3d.view(ST_size, batch_size,
                                                          map_channels - self.dimension_fg, map_width,
                                                          map_width)], dim=2)
        else:
            latent_shuffled = map_from_3d.view(ST_size, batch_size,
                                               map_channels, map_width, map_width)

        output_crop_rotated = self.decoder(ST_flatten(latent_shuffled))
        output_crop_rotated = ST_split(output_crop_rotated)

        ###############################################
        # de-shuffling
        output_crop = torch.index_select(output_crop_rotated, dim=1, index=shuffled_pose_inv)

        if self.masked_blending:
            output_img_crop = output_crop[:, :, 0:3, :, :] #Warning
            mask_enforced_minimum = 0.0001 # HACK 0.05 was used for training the detector, 0.0001 worked for single person
            output_mask_crop = mask_enforced_minimum + (1.0 - mask_enforced_minimum) * F.sigmoid(output_crop[:,:, 3:4, :, :]
                                                                                                 )
            if self.spatial_transformer and detect_encode_decode.bump_function is not None:
                output_mask_crop = output_mask_crop * detect_encode_decode.bump_function.unsqueeze(0).unsqueeze(0)
            if self.scale_mask_max_to_1:
                # do it here already to see maxima in the crop region (debugging), later on it is caled over the whole image again # maximize response map V1 (inside crop, works fine as long as the whole crop is seen)
                # Conclusion: Do it at both places, to get sensible estimates for the mask density. Is it necessary?
                mask_max, max_index = torch.max(features_flatten(output_mask_crop), dim=2)
                # output_mask_crop = output_mask_crop / (0.00001 + mask_max.view(batch_size, 1, 1, 1))
                output_mask_crop = output_mask_crop / (0.0001 + mask_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        else:
            output_img_crop = output_crop

        ###############################################
        # undo spatial transformer
        if self.spatial_transformer:
            # unwarp the decoded image, to be able to optimize the border (window size) we compute blending weights by warping a window of ones with the same function and using that as blending weights with the true BG
            # affine_matrix_uncrop_shuffled = torch.index_select(affine_matrix_uncrop, dim=0, index=shuffled_pose)
            # grid_uncrop_shuffled = F.affine_grid(affine_matrix_uncrop_shuffled, input_img.size())

            # TODO: is there a way to crop multiple times across batches?
            output_imgs = []
            output_masks = []
            mask_densities = []
            input_img_size = input_dict['img'].squeeze().size()

            for j in range(ST_size):
                weights_j = similarity_matrix_normalized[j].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # before, leads to identity flip after rotation
                output_img_crop_j = torch.sum(output_img_crop * weights_j.expand_as(output_img_crop),dim=0)
                grid_uncrop = F.affine_grid(affine_matrix_uncrop_multi[j], input_img_size)
                output_img_warped = F.grid_sample(output_img_crop_j, grid_uncrop, padding_mode='border')
                output_imgs.append(output_img_warped)

                if self.masked_blending:
                    mask = torch.sum(output_mask_crop * weights_j.expand_as(output_mask_crop),dim=0)
                else:
                    mshape = list(output_img_crop[j].size())
                    mshape[1] = 1 # change from 3 channel image to 1 channel mask
                    if detect_encode_decode.bump_function is not None:
                        mask = detect_encode_decode.bump_function.unsqueeze(0).unsqueeze(0).expand(mshape)
                    else:
                        mask = torch.ones(mshape).to(device)
                if self.normalize_mask_density:
                    mask_density = torch.sum(torch.sum(mask,dim=2,keepdim=True), dim=3, keepdim=True)
                    mask_densities.append(mask_density)
                output_mask_warped = F.grid_sample(mask, grid_uncrop, padding_mode='zeros')
                if self.scale_mask_max_to_1:  # maximize response map V2 (after unwarping, ensures to only consider mask pixels inside the output image, not those cropped)
                    mask_max, max_index = torch.max(output_mask_warped.view(batch_size,-1), dim=1)
                    output_mask_warped = output_mask_warped / (0.0001 + mask_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                output_masks.append(output_mask_warped)
            if ST_size > 1:
                output_imgs = torch.stack(output_imgs)
                output_masks = torch.stack(output_masks)
            else: # requires less memory:
                output_imgs = output_imgs[0].unsqueeze(0)
                output_masks = output_masks[0].unsqueeze(0)
                
            if self.predict_transformer_depth:
                #sqrt_2 = float(np.sqrt(2))
                #sqrt_2_by_pi = float(np.sqrt(2 / np.pi))
                opacity_factor = 1 # opacity/float(np.sqrt(2 / np.pi)) # if sqrt(2) etc is included a factor of five was good
                # Note, the following unsqueezing puts different blob positions in dim=0 and sample positoins in dim=1 (they can be the same)
                c = opacity_factor*output_masks.unsqueeze(1)
                # TODO: is this expand needed? Broadcasting should work..
                mu = ST_depth.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                # Note, the sampling position is taken relative to the center of the Gaussians
                mu_offset = 0
                s = ST_depth.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)-mu_offset
                # negative_accumulated_density = sigma * opacity / sqrt_2_by_pi * (torch.erf(0-mu/(sqrt_2*sigma)) - torch.erf(s-mu/(sqrt_2*sigma)))
                # accumulated_density_individual = opacity*sqrt_pi/2 * (torch.erf((s-mu)) - torch.erf((0-mu)) ) # simplified version with sigma = 1/sqrt_2
                # accumulated_density_individual = opacity*sqrt_pi/2 * ( torch.erf((s-mu))) # simplified with origin at -infinity:
                # erf(a,b) = erf(b) - erf(a) => erf(-inf,b) = erf(b) - erf(-inf) = erf(b) + erf(inf) = erf(b)+1  , therefore add the constant term erf()
                accumulated_density_indi = c * ( torch.erf(s-mu)+1) # simplified with c = opacity*sqrt_pi/2
                accumulated_density_sum  = torch.sum(accumulated_density_indi, dim=0) # sum across all blobs (dim=0), individual for each sample point (dim=1)

                # from ECCV 2016 paper (Volumetric contour cues):
                # d_acc = sqrt_2_pi * sum_i(opacity * sigma)
                # d_acc = sqrt_2_pi/sqrt_2 * sum_i( opacity) # sigma = 1/sqrt_2
                # d_acc = sqrt_pi * sum_i(opacity)
                # d_acc = sqrt_pi*(2/sqrt_pi) * sum_i(c) # using c = opacity*sqrt_pi/2 => opacity = c*2/sqrt_pi
                # d_acc = 2 * sum_i(c)
                accumulated_density_bg_indi = 2*opacity_factor*output_masks
                accumulated_density_bg_sum  = torch.sum(accumulated_density_bg_indi, dim=0) # sum across all blobs (dim=0)

                transmittance    = torch.exp(-accumulated_density_sum)
                transmittance_bg = torch.exp(-accumulated_density_bg_sum)
                # TODO, in principle there should be a factor 2/sqrt(pi) here, to model the emmisiion of a gaussian density, but it does not matter due to the used normalization
                #   emission = output_masks*2/float(np.sqrt(2))
                emission = output_masks
                radiance = transmittance * emission
                # note, assuming the emission of the bg is 1 at infinity, hence:
                radiance_bg = transmittance_bg
                radiance_fg_analytic = 1-radiance_bg
                radiance_fg_approx = torch.sum(radiance, dim=0)
                normalization_facor = radiance_fg_analytic/(0.0001+radiance_fg_approx)
                radiance = radiance * normalization_facor # normalized

                max_depth, _ = torch.max(ST_depth,dim=0)
                bg_dist = max_depth+1 # put it 1*sigma=1 behind the last point
                depth_map = (  bg_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*radiance_bg
                             + torch.sum(ST_depth.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * radiance, dim=0)).data.cpu()
                eps = 0.0001
            else:
                # sum over STs and ensure that the sum is less than 1. Otherwise scale each mask down respectively
                if ST_size > 1:
                    mask_sum = torch.sum(output_masks, keepdim=True, dim=0)
                    output_masks /= torch.clamp(mask_sum, min=1, max=9999)
                radiance = output_masks

            radiance_normalized = radiance

            ######## normalize so that having one mask with a low density is peanalized (density is enforced to be the same) #######
            # Note, this has to happen after the unwarping and radiance computation/normalization, first, because we unwarp separately per crop id.
            # Second, because we normalize height on the full crop
            if self.normalize_mask_density:
                mask_densities = torch.stack(mask_densities) # note, this density is normalized by the crop size (not computed in absolute coords)
                min_density, _ = torch.min(mask_densities, dim=0, keepdim=True)
                eps = 0.0001
                radiance_normalized = radiance_normalized*(min_density/(eps+mask_densities))

            if 0 and 'shuffled_appearance_weight' in input_dict.keys():  # HACK disabled!!
                w = input_dict['shuffled_appearance_weight'].item()
                bg = input_dict['bg']
                bg[0,:,:,:] = (1 - w) * bg[0] + w * bg[1]

            # weighted sum over spatial transformer (dimension 0)
            output_img  = torch.sum(radiance_normalized * output_imgs, dim=0)
            output_img += (1 - torch.sum(radiance_normalized, dim=0)) * input_dict['bg']
            del output_imgs

            # undo potential shuffling before color assignment
            radiance_colored = radiance_normalized.cpu()
            radiance_normalized = radiance_normalized.data.cpu()
            
            # move spatial transformer dimension to color dimension, to get semantic segmentation result
            # [0] to remove singleton dim which is now ST, don't use squeeze as it might remove ST_size==1 case too
            radiance_colored = radiance_colored.transpose(0,2)[0]

            if radiance_colored.shape[1]==1 or radiance_colored.shape[1]==3:
                pass
            elif radiance_colored.shape[1]==2:
                radiance_colored = radiance_colored.repeat([1,2,1,1])[:,:3,:,:]
                radiance_colored[:,2,:,:] = 0
            elif radiance_colored.shape[1]>3 and radiance_colored.shape[1]%2==0:
                s = radiance_colored.shape
                radiance_colored = torch.sum(radiance_colored.reshape([s[0],2,-1,s[2],s[3]]),dim=2)
            else: # coloring for odd numbers
                s = radiance_colored.shape
                multiple_of_three = s[1] - (s[1] % 3)
                radiance_colored_t = torch.sum(
                    radiance_colored[:, :multiple_of_three, :, :].data.reshape([s[0], 3, -1, s[2], s[3]]), dim=2)
                radiance_colored_t += torch.sum(radiance_colored[:, multiple_of_three:, :, :].data, dim=1, keepdim=True)
                radiance_colored = torch.clamp(radiance_colored_t.data, 0, 1)

            output_mask_combined = radiance_colored.data.cpu() # drop more than three dimensions
        else:
            if self.masked_blending:
                bg_crop = input_dict['bg_crop']
                output_img = output_mask_crop * output_img_crop + (1 - output_mask_crop) * bg_crop
            else:
                output_img = output_img_crop

        ###############################################
        if 0: # might be slow, therefore disabled in general
            if torch.isnan(output_img).any():
                print('WARNING: unet_encode3D_V2 after decoding: torch.isnan(output_img)')
                print('grid_crop', grid_crop)
                print('grid_uncrop', grid_uncrop)
                IPython.embed()
                eliminateNaNs(output_img)
            if torch.isnan(output_img_crop).any():
                print('WARNING: unet_encode3D_V2 after decoding: torch.isnan(output_img_crop)')
                eliminateNaNs(output_img_crop)
            if torch.isnan(output_mask_crop).any():
                print('WARNING: unet_encode3D_V2 after decoding: torch.isnan(output_mask_crop)')
                eliminateNaNs(output_mask_crop)
            if torch.isnan(output_mask_combined).any():
                print('WARNING: unet_encode3D_V2 after decoding: torch.isnan(output_mask_combined)')
                eliminateNaNs(output_mask_combined)

        ###############################################
        # output stage
        output_dict_all = {'img_crop': ST_flatten(output_img_crop.data.cpu().transpose(1,0).contiguous()), # transpose to make crops from the same image neighbors
                           'img': output_img,
                           'shuffled_pose': shuffled_pose,
                           'shuffled_pose_inv': shuffled_pose_inv,
                           'shuffled_appearance': shuffled_appearance,
                           'latent_3d': ST_flatten(latent_3d),
                           'latent_3d_rotated': latent_3d_rotated,
                           'latent_fg': latent_fg,
                           'cam2cam': cam2cam}  # , 'shuffled_appearance' : xxxx, 'shuffled_pose' : xxx}


        if self.spatial_transformer:
            output_dict_all['spatial_transformer'] = affine_matrix_crop_multi
            output_dict_all['spatial_transformer_raw'] = affine_matrix_crop_raw
            output_dict_all['radiance_normalized'] = radiance_normalized
            output_dict_all['bg_crop'] = ST_flatten(ST_split(input_dict_cropped['bg_crop']).transpose(1,0).contiguous())
            output_dict_all['spatial_transformer_img_crop'] = ST_flatten(ST_split(input_dict_cropped['img_crop']).transpose(1,0).contiguous())
            output_dict_all['img_downscaled'] = input_img_downscaled
            output_dict_all['similarity_matrix'] = similarity_matrix_normalized
            if self.predict_transformer_depth:
                output_dict_all['ST_depth'] = ST_depth
                output_dict_all['depth_map'] = depth_map

            if detect_encode_decode.bump_function is not None:
                output_dict_all['smooth_mask'] = detect_encode_decode.bump_function

            if self.masked_blending:
                output_dict_all['blend_mask'] = output_mask_combined #output_mask_warped

        if self.masked_blending:
            output_dict_all['blend_mask_crop'] = ST_flatten(output_mask_crop.data.cpu().transpose(1,0).contiguous())

        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict

    def forward_pose(self, input_dict):
        if 'img' in input_dict.keys():
            batch_size = input_dict['img'].size()[0]
        else:
            batch_size = input_dict['img_crop'].size()[0]
        ST_size = self.ST_size

        def ST_flatten(batch):
            shape_orig = batch.shape
            #assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = [ST_size*batch_size] + list(shape_orig[2:])
            return batch.view(shape_new)

        def ST_split(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size * batch_size
            shape_new = [ST_size, batch_size] + list(shape_orig[1:])
            return batch.view(shape_new)

        def features_flatten(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1]
            return batch.view(shape_new)

        def features_split3D(batch):
            shape_orig = batch.shape
            assert shape_orig[0] == ST_size and shape_orig[1] == batch_size
            shape_new = list(shape_orig[:2]) + [-1,3]
            return batch.view(shape_new)

        ###############################################
        # spatial transformer
        if self.spatial_transformer:
            input_dict_cropped, input_img_downscaled, ST_depth, affine_matrix_crop_multi, affine_matrix_uncrop_multi, affine_matrix_crop_raw, R_virt2orig = self.forward_detector(input_dict)
        else:
            input_dict_cropped = input_dict  # fallback to using crops from dataloader

        ###############################################
        # encoding stage
        has_fg = hasattr(self, "to_fg")
        output = self.encoder.forward(input_dict_cropped)['latent_3d']
        if has_fg:
            latent_fg = output[:, :self.dimension_fg] #.contiguous().clone() # TODO
            latent_fg = ST_split(latent_fg)
        latent_3d = output[:, self.dimension_fg:self.dimension_fg + self.dimension_3d]
        # make it into a 3D latent space
        #latent_3d = features_split3D(ST_split(latent_3d))

        ###############################################
        # decoding stage
        pose_3d = self.pose_decoder({'latent_3d':latent_3d})['3D']
        pose_3d = ST_split(pose_3d)

        # flip predicted poses to be sorted left to right (mostly for display purposes)
        if ST_size>1:
            crop_x = affine_matrix_crop_multi[:, :, 0, 2]
            is_left_right = ((torch.sign(crop_x[1]-crop_x[0])+1)/2).byte()
            not_left_right = 1 - is_left_right
            mask = torch.stack([is_left_right,not_left_right]).unsqueeze(-1)

            pose_3d_left  = transposed_mask_select(pose_3d,   mask, (0,1))
            pose_3d_right = transposed_mask_select(pose_3d, 1-mask, (0,1))

            pose_3d = torch.stack([pose_3d_left,pose_3d_right]).view(pose_3d.shape)

        ###############################################
        # 3D pose stage (parallel to image decoder)
        output_dict_all = {'3D': pose_3d.transpose(1,0).contiguous().view(batch_size,ST_size,-1,3),
                           'latent_3d': latent_3d,
                           'latent_fg': latent_fg,
                           }
        if self.spatial_transformer:
            output_dict_all['spatial_transformer'] = affine_matrix_crop_multi
            output_dict_all['spatial_transformer_img_crop'] = ST_flatten(ST_split(input_dict_cropped['img_crop']).transpose(1,0).contiguous())
            output_dict_all['spatial_transformer_raw'] = affine_matrix_crop_raw
            output_dict_all['img_downscaled'] = input_img_downscaled
            if self.predict_transformer_depth:
                output_dict_all['ST_depth'] = ST_depth

        output_dict = {}
        for key in self.output_types:
            output_dict[key] = output_dict_all[key]

        return output_dict

# Transmittance, density, visibility DEBUGGING
if 0:
    # import IPython
    # IPython.start_ipython()
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.special import erf
    # opacity_factor = 2/float(np.sqrt(2 / np.pi))
    opacity_factor = 2  # 2 gices 1.8% translucency, 2.5 gives 0.7% translucency to bg
    x = np.linspace(0, 1, 20)
    d = 2 * opacity_factor * x
    t = np.exp(-d)
    plt.plot(x, d, color='r')
    plt.plot(x, t, color='k')
    plt.show()

    # sampling based
    x = np.linspace(0, 3, 50)
    c = opacity_factor * np.array([0.9, 0.5])
    mu = np.array([1, 3])
    d_s = sum([c[i] * np.exp(-(x - m) ** 2) for i, m in enumerate(mu)])
    da_s = sum([c[i] * (erf(x - m) + 1) for i, m in enumerate(mu)])
    da_0 = sum([c[i] * (erf(mu[0] - m) + 1) for i, m in enumerate(mu)])
    da_1 = sum([c[i] * (erf(mu[1] - m) + 1) for i, m in enumerate(mu)])
    # compute transmittance and radiance at sampling points (at the mean positions)
    t_s = np.exp(-da_s)
    t_0 = np.exp(-da_0) * c[0]
    t_1 = np.exp(-da_1) * c[1]
    t_bg = np.exp(-2 * sum([c[i] for i in range(2)]))
    t_0, t_1, t_bg
    normalization_facor = (1 - t_bg) / (0.0001 + sum([t_0, t_1]))
    t_0n = t_0 * normalization_facor
    t_1n = t_1 * normalization_facor
    'raw', t_0, t_1, t_bg, 'normalized', t_0n, t_1n, 'sum(normalized)', sum([t_0n, t_1n, t_bg])
    plt.plot(x, d_s, color='r')
    plt.plot(x, da_s, color='g')
    plt.plot(x, t_s, color='k')
    # plt.plot(x,x*0+t[-1],color='k',ls='-')
    plt.show()
