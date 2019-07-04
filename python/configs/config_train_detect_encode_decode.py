from utils import skeleton

# problem class parameters
numJoints = 17
inputDimension = 128

config_dict = {
    # general params
    'dpi' : 300,
    #'config_class_file': 'dict_configs/config_class_encodeDecode.py',
    'input_types'       : ['img','bg','R_cam_2_world'], # ,'iteration'
    'output_types'      : ['img','img_crop','bg_crop','spatial_transformer','img_downscaled','blend_mask','blend_mask_crop',
                           'similarity_matrix','spatial_transformer_img_crop','latent_fg','latent_3d',
                           'shuffled_pose','shuffled_pose_inv','shuffled_appearance',
                           'radiance_normalized','ST_depth','depth_map'], #,'smooth_mask',
    'label_types_train' : ['img','3D'],#'R_world_2_cam','R_cam_2_world'],
    'label_types_test'  : ['img','3D'],#'R_world_2_cam','R_cam_2_world'],
    'num_workers'       : 0, # HACK 8

    # problem class parameters
    'bones' : skeleton.bones_h36m,

    # opt parameters
    'num_training_iterations' : 600000,
    'save_every' : 100000,
    'learning_rate' : 1e-3,# baseline: 0.001=1e-3
    'test_every' : 5000,
    'plot_every' : 5000,
    'print_every' : 100,

    # network parameters
    'batch_size_train' : 16,
    'batch_size_test' : 16, #10 #self.batch_size # Note, needs to be = self.batch_size for multi-view validation
    'outputDimension_3d' : numJoints * 3,
    'outputDimension_2d' : inputDimension // 8,

    # loss
    'train_scale_normalized' : True,
    'train_crop_relative' : False,

    # dataset
    'dataset_folder_train' : '/cvlabdata1/home/rhodin/datasets/EPFL-AmateurBoxingDataset/EPFL-AmateurBoxingDataset-train',
    'dataset_folder_test' : '/cvlabdata1/home/rhodin/datasets/EPFL-AmateurBoxingDataset/EPFL-AmateurBoxingDataset-val',
    'img_type' : 'jpg',
    'fullFrameResolution': [910, 512],
    'img_mean' : (0.485, 0.456, 0.406),
    'img_std' : (0.229, 0.224, 0.225),
    'actor_subset' : None, #[1,5,6,7,8], # all training subjects
    'active_cameras' : False,
    'inputDimension' : inputDimension,
    'mirror_augmentation' : False,
    'perspectiveCorrection' : True,
    'rotation_augmentation' : True,
    'shear_augmentation' : 0,
    'scale_augmentation' : False,
    'seam_scaling' : 1.0,
    'useCamBatches' : 2,
    'useSubjectBatches' : True,
    'every_nth_frame' : 1,

    'note' : 'resL3',

    # encode decode
    'latent_bg' : 0,
    'latent_fg' : 128,
    'latent_3d' : 200*3,
    'latent_dropout' : 0.3,
    'from_latent_hidden_layers' : 0,
    'upsampling_bilinear' : 'upper',
    'shuffle_fg' : True,
    'shuffle_3d' : True,
    'feature_scale' : 2,
    'num_encoding_layers' : 4,
    'loss_weight_rgb' : 1,
    'loss_weight_gradient' : 0,
    'loss_weight_imageNet' : 2,
    'loss_weight_3d' : 0,
    'do_maxpooling' : True,
    'implicit_rotation' : False,
    'predict_rotation' : False,
    'skip_background' : True,

    'training_mode' : "NVS",

    'spatial_transformer': 'GaussBSqSqr',  # True,
    'spatial_transformer_num': 2,  # True,
    'spatial_transformer_bounds': {'border_factor': 0.8, 'min_size': 0.2, 'max_size': 1},
    'masked_blending': True,
    'scale_mask_max_to_1': True,
    'predict_transformer_depth': True,  # Note, disable switch later on
    'pass_transformer_depth': False,
    'normalize_mask_density': False,
    'match_crops': True,
    'offset_crop': False,  # jitter
    'transductive_training': [],
}

config_dict['batch_size_train'] = 16
config_dict['batch_size_test'] = 8

# shuffling accross time
if 1:
    config_dict['shuffle_fg'] = False
# classic auto encoder, with some billinear layers
if 0:
    config_dict['shuffle_3d'] = False

# no appearance
if 0:
    config_dict['latent_fg'] = 0

# test more than necessary number of transformers
if 0:
    config_dict['spatial_transformer_num'] = 3  # True,
    config_dict['batch_size_train'] = 12 # needed for boxing
    config_dict['similarity_bandwidth'] = 2

# smaller unsupervised subsets
if 0:
    config_dict['actor_subset'] = [1,5,6,7,8]
    #config_dict['actor_subset'] = [1,5,6]
    config_dict['actor_subset'] = [1]

network_path = '../output/train_detectNVS_resL3_layers4_wRGB1_wGrad0_wImgNet2_fg128_ldrop0o3_billinupper_fscale2_shuffleFGFalse_shuffle3dTrue_nth1_cFalse_subNone_bs2_lr0o001_'
#config_dict['pretrained_detector_path'] = network_path + '/models/network_last_val.pth'

network_path = '../output/train_detectNVS_e40'
#config_dict['pretrained_network_path'] = network_path + '/models/network_last_val.pth'
