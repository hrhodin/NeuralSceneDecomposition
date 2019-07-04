from utils import io as utils_io
import os

if 0:
    config_dict = utils_io.loadModule("./configs/config_train_detect_encode_decode.py").config_dict
    network_name = '/network_NVS_best.pth'
else:
    config_dict = utils_io.loadModule("./configs/config_train_detect_encode_decode_pose.py").config_dict
    network_name = '/network_pose_best.pth'

config_dict['num_workers'] = 0
config_dict['label_types_test'].remove('img')
config_dict['label_types_train'].remove('img')
config_dict['batch_size_train'] = 4
config_dict['batch_size_test'] = 4

if 0:
    network_path = '../output/train_detectNVS_resL3_layers4_wRGB1_wGrad0_wImgNet2_fg128_ldrop0o3_billinupper_fscale2_shuffleFGFalse_shuffle3dTrue_nth1_cFalse_subNone_bs2_lr0o001_'
    config_dict['pretrained_network_path'] = network_path + '/models/network_best_val_t1.pth'
else:
    network_path = '../examples'
    config_dict['pretrained_network_path'] = network_path + network_name
    if not os.path.exists(config_dict['pretrained_network_path']):
        import urllib.request
        print("Downloading pre-trained weights, can take a while...")
        urllib.request.urlretrieve("https://datasets-cvlab.epfl.ch/RhodinCVPR2019/"+network_name,
                                   config_dict['pretrained_network_path'])
        print("Downloading done.")
