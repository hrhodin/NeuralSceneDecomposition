import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from datasets import collected_dataset

import numpy as np
import IPython

from utils import io as utils_io
from utils import datasets as utils_data
from utils import training as utils_train
from utils import plot_dict_batch as utils_plot_batch

from models import detect_encode_decode
from losses import generic as losses_generic
from losses import images as losses_images

import math
import torch
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models_tv

import sys
sys.path.insert(0,'./ignite')
from ignite._utils import convert_tensor
from ignite.engine import Events

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

class IgniteTrainNVS:
    def run(self, config_dict_file, config_dict):
        # create visualization windows
        try:
            import visdom
            port = 3557
            vis = visdom.Visdom(port=port)
            if not vis.check_connection():
                vis = None
                print("WARNING: Visdom server not running. Please run 'python -m visdom.server -port port' to see visual output")
            else:
                print("Visdom connected, reporting progress there!")
        except ImportError:
            vis = None
            print("WARNING: No visdom package is found. Please install it with command: \n pip install visdom to see visual output")
            #raise RuntimeError("WARNING: No visdom package is found. Please install it with command: \n pip install visdom to see visual output")
        vis_windows = {}
    
        # save path and config files
        save_path = self.get_parameter_description(config_dict)
        utils_io.savePythonFile(config_dict_file, save_path)
        utils_io.savePythonFile(__file__, save_path)
        
        # now do training stuff
        epochs = 200
        train_loader = self.load_data_train(config_dict)
        test_loader = self.load_data_test(config_dict)
        model = self.load_network(config_dict)
        model = model.to(device)
        optimizer = self.loadOptimizer(model,config_dict)
        loss_train,loss_test = self.load_loss(config_dict)
            
        trainer = utils_train.create_supervised_trainer(model, optimizer, loss_train, device=device)
        evaluator = utils_train.create_supervised_evaluator(model,
                                                metrics={#'accuracy': CategoricalAccuracy(),
                                                         'primary': utils_train.AccumulatedLoss(loss_test)},
                                                device=device)
    
        #@trainer.on(Events.STARTED)
        def load_previous_state(engine):
            utils_train.load_previous_state(save_path, model, optimizer, engine.state)
             
        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_progress(engine):
            # log the loss
            iteration = engine.state.iteration - 1
            if iteration % config_dict['print_every'] == 0:
                utils_train.save_training_error(save_path, engine, vis, vis_windows)
        
            # log batch example image
            #if iteration in [0,100,500,1000,2000,5000,10000,20000,50000,100000,200000]:
            if iteration in [0,100,500,1000,2000,] or iteration % config_dict['plot_every'] == 0:
                utils_train.save_training_example(save_path, engine, vis, vis_windows, config_dict)
                
        #@trainer.on(Events.EPOCH_COMPLETED)
        @trainer.on(Events.ITERATION_COMPLETED)
        def validate_model(engine):
            iteration = engine.state.iteration - 1
            if (iteration+1) % config_dict['test_every'] != 0: # +1 to prevent evaluation at iteration 0
                return
            print("Running evaluation at iteration",iteration)
            evaluator.run(test_loader)
            avg_accuracy = utils_train.save_testing_error(save_path, engine, evaluator, vis, vis_windows)
    
            # save the best model
            utils_train.save_model_state(save_path, trainer, avg_accuracy, model, optimizer, engine.state)
    
        # print test result
        @evaluator.on(Events.ITERATION_COMPLETED)
        def log_test_loss(engine):
            iteration = engine.state.iteration - 1
            if iteration in [0,100]:
                utils_train.save_test_example(save_path, trainer, evaluator, vis, vis_windows, config_dict)
    
        # kick everything off
        trainer.run(train_loader, max_epochs=epochs)
        
    def load_network(self, config_dict):
        output_types= config_dict['output_types']
        
        use_billinear_upsampling = config_dict.get('upsampling_bilinear', False)
        lower_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'half'
        upper_billinear = 'upsampling_bilinear' in config_dict.keys() and config_dict['upsampling_bilinear'] == 'upper'

        if lower_billinear:
            use_billinear_upsampling = False
        network_single = detect_encode_decode.detect_encode_decode(dimension_bg=config_dict['latent_bg'],
                                            dimension_fg=config_dict['latent_fg'],
                                            dimension_3d=config_dict['latent_3d'],
                                            feature_scale=config_dict['feature_scale'],
                                            shuffle_fg=config_dict['shuffle_fg'],
                                            shuffle_3d=config_dict['shuffle_3d'],
                                            latent_dropout=config_dict['latent_dropout'],
                                            in_resolution=config_dict['inputDimension'],
                                            is_deconv=not use_billinear_upsampling,
                                            upper_billinear=upper_billinear,
                                            lower_billinear=lower_billinear,
                                            num_encoding_layers=config_dict.get('num_encoding_layers', 4),
                                            output_types=output_types,
                                            subbatch_size=config_dict['useCamBatches'],
                                            implicit_rotation=config_dict['implicit_rotation'],

                                            mode=config_dict['training_mode'],
                                            spatial_transformer=config_dict.get('spatial_transformer', False),
                                            ST_size=config_dict.get('spatial_transformer_num', 1),
                                            spatial_transformer_bounds=config_dict.get('spatial_transformer_bounds', {'border_factor':1, 'min_size':0.1, 'max_size':1}),
                                            masked_blending=config_dict.get('masked_blending',True),
                                            scale_mask_max_to_1=config_dict.get('scale_mask_max_to_1',True),
                                            predict_transformer_depth=config_dict.get('predict_transformer_depth',False),
                                            normalize_mask_density=config_dict.get('normalize_mask_density',False),
                                            match_crops=config_dict.get('match_crops',False),
                                            offset_crop=config_dict.get('offset_crop',False),
                                            similarity_bandwidth=config_dict.get('similarity_bandwidth',10),
                                            disable_detector=config_dict.get('disable_detector',False),
                                            )

        if 'pretrained_network_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_network_path']")
            pretrained_network_path = config_dict['pretrained_network_path']
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0) # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_network_path']")
        
        if 'pretrained_detector_path' in config_dict.keys(): # automatic
            print("Loading weights from config_dict['pretrained_detector_path']")
            pretrained_network_path = config_dict['pretrained_detector_path']
            pretrained_states = torch.load(pretrained_network_path, map_location=device)
            utils_train.transfer_partial_weights(pretrained_states, network_single, submodule=0, prefix='detector') # last argument is to remove "network.single" prefix in saved network
            print("Done loading weights from config_dict['pretrained_detector_path']")
        return network_single
    
    def loadOptimizer(self,network, config_dict):
        params_all_id = list(map(id, network.parameters()))
        params_encoder_id = list(map(id, network.encoder.parameters()))
        params_encoder_finetune_id = [] \
                                     + list(map(id, network.encoder.layer4_reg.parameters())) \
                                     + list(map(id, network.encoder.layer3.parameters())) \
                                     + list(map(id, network.encoder.l4_reg_toVec.parameters())) \
                                     + list(map(id, network.encoder.fc.parameters()))

        params_decoder_id = list(map(id, network.decoder.parameters()))
        params_detector_id = list(map(id, network.detector.parameters()))

        params_except_encode_decode = [id for id in params_all_id if id not in params_decoder_id + params_encoder_id]
        params_except_detect_encode_decode = [id for id in params_all_id if
                                              id not in params_decoder_id + params_encoder_id + params_detector_id]

        # for the more complex setup
        # if False: # after many iterations it still diverges, at least with Huber and L1 loss 'pretrained_detector_path' in self.config_dict or 'pretrained_network_path' in self.config_dict: # NO, smoothness is not sufficient: self.config_dict['spatial_transformer'] == 'GaussBSqSqr':
        #    params_normal_id = params_except_encode_decode + params_encoder_finetune_id + params_decoder_id
        #    params_slow_id = [] # with pre-trained detector it should be fine to run at full lr
        if config_dict.get('fix_detector_weight', False):
            params_normal_id = params_except_detect_encode_decode + params_encoder_finetune_id
            params_slow_id = params_decoder_id  # used to slow down decoder, less ceivir but still necessary after removal of batch norm
        else:
            params_normal_id = params_except_encode_decode + params_encoder_finetune_id
            params_slow_id = params_decoder_id  # used to slow down decoder, less ceivir but still necessary after removal of batch norm

        params_normal = [p for p in network.parameters() if id(p) in params_normal_id]
        params_slow = [p for p in network.parameters() if id(p) in params_slow_id]
        params_static_id = [id_p for id_p in params_all_id if not id_p in params_normal_id + params_slow_id]

        # disable gradient computation for static params, saves memory and computation
        for p in network.parameters():
            if id(p) in params_static_id:
                p.requires_grad = False

        print("Normal learning rate: {} params".format(len(params_normal_id)))
        print("Slow learning rate: {} params".format(len(params_slow)))
        print("Static learning rate: {} params".format(len(params_static_id)))
        print("Total: {} params".format(len(params_all_id)), 'sum of all ',
              len(params_normal_id) + len(params_slow) + len(params_static_id))

        self.opt_params = [
            {'params': params_normal,
             'lr': config_dict['learning_rate']},
            {'params': params_slow,
             'lr': config_dict['learning_rate'] / 5}
            # lr=1/2 worked for view change probability =0.5, with 0.75 is diverged
        ]

        optimizer = torch.optim.Adam(self.opt_params, lr=config_dict['learning_rate'])  # weight_decay=0.0005
        return optimizer
    
    def load_data_train(self,config_dict):
        dataset = collected_dataset.CollectedDataset(data_folder=config_dict['dataset_folder_train'], img_type=config_dict['img_type'],
            input_types=config_dict['input_types'], label_types=config_dict['label_types_train'])

        batch_sampler = collected_dataset.CollectedDatasetSampler(data_folder=config_dict['dataset_folder_train'],
              actor_subset=config_dict['actor_subset'],
              useSubjectBatches=config_dict['useSubjectBatches'], useCamBatches=config_dict['useCamBatches'],
              batch_size=config_dict['batch_size_train'],
              randomize=True,
              every_nth_frame=config_dict['every_nth_frame'])

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                             collate_fn=utils_data.default_collate_with_string)
        return loader
    
    def load_data_test(self,config_dict):
        dataset = collected_dataset.CollectedDataset(data_folder=config_dict['dataset_folder_test'], img_type=config_dict['img_type'],
            input_types=config_dict['input_types'], label_types=config_dict['label_types_test'])

        batch_sampler = collected_dataset.CollectedDatasetSampler(data_folder=config_dict['dataset_folder_test'],
            useSubjectBatches=0, useCamBatches=config_dict['useCamBatches'],
            batch_size=config_dict['batch_size_test'],
            randomize=True,
            every_nth_frame=100) #config_dict['every_nth_frame'])

        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=config_dict['num_workers'], pin_memory=False,
                                             collate_fn=utils_data.default_collate_with_string)

        if 0: # save data for demo
            import pickle
            data_iterator = iter(loader)
            data_input, data_labels = next(data_iterator) #[next(data_iterator) for i in range(3)]
            batch_size = 8
            input = {'img': np.array(data_input['img'][:batch_size].numpy(), dtype='float16'),
                     'bg': np.array(data_input['bg'][:batch_size].numpy(), dtype='float16'),
                     'R_cam_2_world': np.array(data_input['R_cam_2_world'][:batch_size].numpy(), dtype='float16'),
                     }
            label = {'3D': np.array(data_labels['3D'][:batch_size].numpy(), dtype='float16'),
                     'pose_mean' : np.array(data_labels['pose_mean'][:batch_size].numpy(), dtype='float16'),
                     'pose_std' : np.array(data_labels['pose_std'][:batch_size].numpy(), dtype='float16')
                     }
            data_cach = tuple([input, label])
            pickle.dump(data_cach, open('../examples/test_set.pickl', "wb"))
            IPython.embed()
            exit()

        return loader
    
    def load_loss(self, config_dict):
        # normal
        if config_dict.get('MAE', False):
            pairwise_loss = torch.nn.modules.loss.L1Loss()
        else:
            pairwise_loss = torch.nn.modules.loss.MSELoss()

        if 1 : #"box" in config_dict['training_set'] or "walk_full" in config_dict['training_set']:
            pairwise_loss = losses_generic.LossInstanceMeanStdFromLabel(pairwise_loss)

        img_key = 'img'
        image_pixel_loss = losses_generic.LossOnDict(key=img_key, loss=pairwise_loss)
        image_imgNet_bare = losses_images.ImageNetCriterium(criterion=pairwise_loss, weight=config_dict['loss_weight_imageNet'], do_maxpooling=config_dict.get('do_maxpooling',True))
        image_imgNet_loss = losses_generic.LossOnDict(key=img_key, loss=image_imgNet_bare)
        
        losses_train = []
        losses_test = []
        
        if img_key in config_dict['output_types']:
            if config_dict['loss_weight_rgb']>0:
                losses_train.append(image_pixel_loss)
                losses_test.append(image_pixel_loss)
            if config_dict['loss_weight_imageNet']>0:
                losses_train.append(image_imgNet_loss)
                losses_test.append(image_imgNet_loss)

        # priors on crop
        if config_dict['spatial_transformer']:
            losses_train.append(losses_generic.AffineCropPositionPrior(config_dict['fullFrameResolution'],weight=0.1))

                
        loss_train = losses_generic.PreApplyCriterionListDict(losses_train, sum_losses=True)
        loss_test  = losses_generic.PreApplyCriterionListDict(losses_test,  sum_losses=True)
                
        # annotation and pred is organized as a list, to facilitate multiple output types (e.g. heatmap and 3d loss)
        return loss_train, loss_test
    
    def get_parameter_description(self, config_dict):#, config_dict):
        folder = "../output/train_detectNVS_{note}_layers{num_encoding_layers}_wRGB{loss_weight_rgb}_wGrad{loss_weight_gradient}_wImgNet{loss_weight_imageNet}_fg{latent_fg}_ldrop{latent_dropout}_billin{upsampling_bilinear}_fscale{feature_scale}_shuffleFG{shuffle_fg}_shuffle3d{shuffle_3d}_nth{every_nth_frame}_c{active_cameras}_sub{actor_subset}_bs{useCamBatches}_lr{learning_rate}_".format(**config_dict)
        folder = folder.replace(' ','').replace('../','[DOT_SHLASH]').replace('.','o').replace('[DOT_SHLASH]','../').replace(',','_')
        #config_dict['storage_folder'] = folder
        return folder
        
    
if __name__ == "__main__":
    config_dict_module = utils_io.loadModule("configs/config_train_detect_encode_decode.py")
    config_dict = config_dict_module.config_dict
    ignite = IgniteTrainNVS()
    ignite.run(config_dict_module.__file__, config_dict)