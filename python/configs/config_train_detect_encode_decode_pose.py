from utils import io as utils_io

#network_path = '/cvlabdata1/home/rhodin/code/humanposeannotation/python/pytorch_selfsupervised_multiview/output/encode_resL3_ResNet_layers4_implRFalse_s3Dp[1_9_11]_w3Dp0_w3D0_wRGB1_wGrad0_wImgNet2_skipBG0_fg128_3dTrue_lh3Dp2_ldrop0o3_billinupper_fscale2_shuffleFGTrue_shuffle3dTrue_h36m_cross_nth1_cFalse_sub[1_5_6_7_8]_bs2_lr0o001_'
if 0:
    network_path = '../output/train_detectNVS_resL3_layers4_wRGB1_wGrad0_wImgNet2_fg128_ldrop0o3_billinupper_fscale2_shuffleFGFalse_shuffle3dTrue_nth1_cFalse_subNone_bs2_lr0o001_'
    config_dict = utils_io.loadModule(network_path + "/config_train_detect_encode_decode.py").config_dict
    config_dict['pretrained_network_path'] = network_path + '/models/network_best_val_t1.pth'
    #config_dict['pretrained_network_path'] = network_path + '/models/network_last_val.pth'
else:
    import os
    config_dict = utils_io.loadModule("configs/config_train_detect_encode_decode.py").config_dict
    config_dict['pretrained_network_path'] = '../examples/network_NVS.pth'
    if not os.path.exists(config_dict['pretrained_network_path']):
        import urllib.request
        print("Downloading pre-trained weights, can take a while...")
        urllib.request.urlretrieve("https://datasets-cvlab.epfl.ch/RhodinCVPR2019/network_NVS.pth",
                                   config_dict['pretrained_network_path'])
        print("Downloading done.")

config_dict['training_mode'] = "pose"

config_dict['label_types_test']  += ['3D','pose_mean','pose_std']
config_dict['label_types_train'] += ['3D','pose_mean','pose_std']
config_dict['output_types'] = ['3D','spatial_transformer']

config_dict['latent_dropout'] = 0

config_dict['shuffle_fg'] = False
config_dict['shuffle_3d'] = False
config_dict['useCamBatches'] = 0
config_dict['useSubjectBatches'] = 0
config_dict['train_scale_normalized'] = 'mean_std'

config_dict['num_workers'] = 0
# pose training on full dataset
#config_dict['actor_subset'] = [9,11]

config_dict['test_every'] = 1000

