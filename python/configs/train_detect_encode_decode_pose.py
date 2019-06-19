import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from datasets import collected_dataset
import sys, os, shutil

from utils import io as utils_io
import numpy as np
import torch
import torch.optim
#import pickle
import IPython

import train_detect_encode_decode
from losses import generic as losses_generic
from losses import poses as losses_poses

class IgniteTrainPose(train_detect_encode_decode.IgniteTrainNVS):
    def loadOptimizer(self, network, config_dict):
        params_all_id = list(map(id, network.parameters()))
        params_posenet_id = list(map(id, network.pose_decoder.parameters()))
        params_toOptimize = [p for p in network.parameters() if id(p) in params_posenet_id]

        params_static_id = [id_p for id_p in params_all_id if not id_p in params_posenet_id]

        # disable gradient computation for static params, saves memory and computation
        for p in network.parameters():
            if id(p) in params_static_id:
                p.requires_grad = False

        print("Normal learning rate: {} params".format(len(params_posenet_id)))
        print("Static learning rate: {} params".format(len(params_static_id)))
        print("Total: {} params".format(len(params_all_id)))

        opt_params = [{'params': params_toOptimize, 'lr': config_dict['learning_rate']}]
        optimizer = torch.optim.Adam(opt_params, lr=config_dict['learning_rate']) #weight_decay=0.0005
        return optimizer

    def load_loss(self, config_dict):
        pose_key = '3D'
        #loss_train = losses_generic.LossLabelMeanStdNormalized(pose_key, torch.nn.MSELoss())
        #loss_test = losses_generic.LossLabelMeanStdUnNormalized(pose_key, losses_poses.Criterion3DPose_leastQuaresScaled(losses_poses.MPJPECriterion(weight=1)), scale_normalized=False)

        loss_train = torch.nn.MSELoss()
        reduction = 'none'
        loss_test = losses_poses.MPJPECriterion(weight=1,reduction=reduction)

        loss_train = losses_poses.StaticDenormalizedLoss(pose_key, loss_train)
        loss_test = losses_generic.LossLabelMeanStdUnNormalized(pose_key, loss_test, scale_normalized=False)

        #if config_dict['spatial_transformer_num'] >= 2:
        #    loss_train = losses_poses.MultiPersonSimpleLoss(loss_train) # minimum of two
        #    loss_test = losses_poses.MultiPersonSimpleLoss(loss_test) # minimum of two

        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test

    def get_parameter_description(self, config_dict):#, config_dict):
        folder = "../output/trainPose_{note}_layers{num_encoding_layers}_implR{implicit_rotation}_fg{latent_fg}_3d{skip_background}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_nth{every_nth_frame}_c{active_cameras}_sub{actor_subset}_bs{useCamBatches}_lr{learning_rate}_".format(**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        return folder

if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_train_detect_encode_decode_pose.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTrainPose()
    ignite.run(config_dict_module.__file__, config_dict)