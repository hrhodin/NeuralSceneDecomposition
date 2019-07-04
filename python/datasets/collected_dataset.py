import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import csv
import numpy as np
import torch
import torchvision
import torch.utils.data as data

import h5py
import imageio

from random import shuffle

import IPython

#import numpy.linalg as la

from utils import datasets as utils_data
from tqdm import tqdm

class CollectedDataset(data.Dataset):
    def __init__(self, data_folder, img_type,
                 input_types, label_types,
                 mean=(0.485, 0.456, 0.406),
                 stdDev= (0.229, 0.224, 0.225),
                 useSequentialFrames=0,
                 ):
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        class Image256toTensor(object):
            def __call__(self, pic):
                img = torch.from_numpy(pic.transpose((2, 0, 1))).float()
                img = img.div(255)
                return img

            def __repr__(self):
                return self.__class__.__name__ + '()'

        self.transform_in = torchvision.transforms.Compose([
            Image256toTensor(), #torchvision.transforms.ToTensor() the torchvision one behaved differently for different pytorch versions, hence the custom one..
            torchvision.transforms.Normalize(self.mean, self.stdDev)
        ])

        h5_label_file = h5py.File(data_folder + '/labels.h5', 'r')
        print('Loading h5 label file to memory')
        self.label_dict = {key: np.array(value) for key, value in h5_label_file.items()}
        print('Loaded {} frames'.format(len(self.label_dict)))

        # Note, after magic number and bone length normalization!
        self.pose_mean = np.array([
                    [ 0.0000,  0.0000,  0.0000],
                    [ 0.0016,  0.1304,  0.0262],
                    [ 0.0055,  0.4796,  0.0955],
                    [ 0.0090,  0.8197,  0.1632],
                    [ 0.0013,  0.1265,  0.0255],
                    [ 0.0057,  0.4444,  0.0895],
                    [ 0.0103,  0.8180,  0.1644],
                    [-0.0026, -0.2240, -0.0449],
                    [-0.0047, -0.4046, -0.0810],
                    [-0.0063, -0.5316, -0.1060],
                    [-0.0080, -0.6843, -0.1362],
                    [-0.0052, -0.4222, -0.0845],
                    [-0.0027, -0.2450, -0.0489],
                    [-0.0045, -0.3044, -0.0606],
                    [-0.0044, -0.4119, -0.0825],
                    [-0.0016, -0.2349, -0.0464],
                    [-0.0022, -0.3627, -0.0711]])
        self.pose_std = np.array(
                   [[1.0000, 1.0000, 1.0000], # Note, edited to downweight first entry..
                    [0.0589, 0.0183, 0.0572],
                    [0.1663, 0.0472, 0.1629],
                    [0.2464, 0.0792, 0.2453],
                    [0.0628, 0.0189, 0.0616],
                    [0.2051, 0.0638, 0.2033],
                    [0.2584, 0.0767, 0.2561],
                    [0.0266, 0.0210, 0.0273],
                    [0.0518, 0.0402, 0.0531],
                    [0.0964, 0.0657, 0.0972],
                    [0.1418, 0.0904, 0.1419],
                    [0.1280, 0.0672, 0.1260],
                    [0.2343, 0.1073, 0.2314],
                    [0.2687, 0.1886, 0.2660],
                    [0.1372, 0.0640, 0.1356],
                    [0.2249, 0.0959, 0.2216],
                    [0.2460, 0.1514, 0.2442]])

    def __len__(self):
        return len(self.label_dict['frame'])

    def getLocalIndices(self, index):
        input_dict = {}
        cam = int(self.label_dict['cam'][index].item())
        seq = int(self.label_dict['seq'][index].item())
        frame = int(self.label_dict['frame'][index].item())
        return cam, seq, frame

    def __getitem__(self, index):
        cam, seq, frame = self.getLocalIndices(index)
        def getImageName(key):
            if key in ['bg']:
                return self.data_folder + '/seq_{:03d}/cam_{:02d}/{}.{}'.format(seq, cam, key, self.img_type)
            else:
                return self.data_folder + '/seq_{:03d}/cam_{:02d}/{}_{:06d}.{}'.format(seq, cam, key, frame, self.img_type)
        def loadImage(name):
            #             if not os.path.exists(name):
            #                 raise Exception('Image not available ({})'.format(name))
            return np.array(self.transform_in(imageio.imread(name)), dtype='float32')
        def loadData(types):
            new_dict = {}
            for key in types:
                if key in ['pose_mean']:
                    new_dict[key] = np.array([self.pose_mean, self.pose_mean], dtype='float32')
                elif key in ['pose_std']:
                    new_dict[key] = np.array([self.pose_std, self.pose_std], dtype='float32')
                elif key in ['img','bg','img_crop','bg_crop']:
                    new_dict[key] = loadImage(getImageName(key)) #np.array(self.transform_in(imageio.imread(getImageName(key))), dtype='float32')
                else:
                    new_dict[key] = np.array(self.label_dict[key][index], dtype='float32')
            return new_dict
        return loadData(self.input_types), loadData(self.label_types)

class CollectedDatasetSampler(data.sampler.Sampler):
    def __init__(self, data_folder, batch_size,
                 actor_subset=None,
                 useSubjectBatches=0, useCamBatches=0,
                 randomize=True,
                 useSequentialFrames=0,
                 every_nth_frame=1):
        # save function arguments
        for arg,val in list(locals().items()):
            setattr(self, arg, val)

        # build cam/subject datastructure
        h5_label_file = h5py.File(data_folder + '/labels.h5', 'r')
        print('Loading h5 label file to memory')
        label_dict = {key: np.array(value) for key, value in h5_label_file.items()}
        self.label_dict = label_dict
        print('Establishing sequence association. Available labels:', list(label_dict.keys()))
        all_keys = set()
        camsets = {}
        sequence_keys = {}
        data_length = len(label_dict['frame'])
        with tqdm(total=data_length) as pbar:
            for index in range(data_length):
                pbar.update(1)
                #if len(label_dict['subj'][index]) == 1:
                #    sub_i = int(label_dict['subj'][index].item())
                #else:
                subjects_i = tuple(label_dict['subj'][index]) #.tolist()
                cam_i = int(label_dict['cam'][index].item())
                seq_i = int(label_dict['seq'][index].item())
                frame_i = int(label_dict['frame'][index].item())

                if actor_subset is not None and any([True for s in subjects_i if int(s) not in actor_subset]):
                    continue

                key = (subjects_i, seq_i, frame_i)
                if key not in camsets:
                    camsets[key] = {}
                camsets[key][cam_i] = index

                # only add if accumulated enough cameras
                if len(camsets[key]) >= self.useCamBatches:
                    all_keys.add(key)

                    if seq_i not in sequence_keys:
                        sequence_keys[seq_i] = set()
                    sequence_keys[seq_i].add(key)

        self.all_keys = list(all_keys)
        self.camsets = camsets
        self.sequence_keys = {seq: list(keyset) for seq, keyset in sequence_keys.items()}
        print("DictDataset: Done initializing, listed {} camsets ({} frames) and {} sequences".format(
                                            len(self.camsets), data_length, len(sequence_keys)))

    def __iter__(self):
        index_list = []
        print("Randomizing dataset (CollectedDatasetSampler.__iter__)")
        with tqdm(total=len(self.all_keys)//self.every_nth_frame) as pbar:
            for index in range(0,len(self.all_keys), self.every_nth_frame):
                pbar.update(1)
                key = self.all_keys[index]
                if 1:
                    subjects_i, seq_i, frame_i = key
                    camset = self.camsets[key]
                    cam_keys = list(camset.keys())
                    for cam_i in cam_keys:
                        bg_file_name = self.getBackgroundName(seq_i, cam_i)
                        if not os.path.exists(bg_file_name):
                            self.computeAndCacheBG(seq_i, cam_i)

                def getCamSubbatch(key):
                    camset = self.camsets[key]
                    cam_keys = list(camset.keys())
                    assert self.useCamBatches <= len(cam_keys)
                    if self.randomize:
                        shuffle(cam_keys)
                    if self.useCamBatches == 0:
                        cam_subset_size = 99
                    else:
                        cam_subset_size = self.useCamBatches
                    cam_indices = [camset[k] for k in cam_keys[:cam_subset_size]]
                    return cam_indices

                index_list = index_list + getCamSubbatch(key)
                if self.useSubjectBatches:
                    seqi = key[1]
                    potential_keys = self.sequence_keys[seqi]
                    key_other = potential_keys[np.random.randint(len(potential_keys))]
                    index_list = index_list + getCamSubbatch(key_other)

        subject_batch_factor = 1+int(self.useSubjectBatches > 0) # either 1 or 2
        cam_batch_factor = max(1,self.useCamBatches)
        sub_batch_size = cam_batch_factor*subject_batch_factor
        assert len(index_list) % sub_batch_size == 0
        indices_batched = np.array(index_list).reshape([-1,sub_batch_size])
        if self.randomize:
            indices_batched = np.random.permutation(indices_batched)
        indices_batched = indices_batched.reshape([-1])[:(indices_batched.size//self.batch_size)*self.batch_size] # drop last frames
        return iter(indices_batched.reshape([-1,self.batch_size]))

    def getBackgroundName(self, seqi, cami):
        return self.data_folder + '/seq_{:03d}/cam_{:02d}/{}.{}'.format(seqi, cami, 'bg', 'jpg')

    def computeAndCacheBG(self, seqi, cami):
        bg_file_name = self.getBackgroundName(seqi, cami)
        bg_path = '/'.join(bg_file_name.split('/')[:-1])

        num_samples = 50
        import os
        names = [os.path.join(bg_path, file) for file in os.listdir(bg_path)]
        names_subsampled = names[0::len(names)//num_samples]
        image_batch = [np.array(imageio.imread(name), dtype='float32') for name in names_subsampled]
        image_batch = np.array(image_batch)
        print("Computing median of {} images".format(len(image_batch)))
        image_median = np.median(image_batch, axis=0)
        imageio.imsave(bg_file_name, image_median)
        print("Saved background image to {}".format(bg_file_name))

# training: 87395
# validation:
# testing: 28400

if __name__ == '__main__':
    dataset = CollectedDataset(
                 data_folder='/Users/rhodin/H36M-MultiView-test',
                 input_types=['img_crop','bg_crop'], label_types=['3D'])

    batch_sampler = CollectedDatasetSampler(
                 data_folder='/Users/rhodin/H36M-MultiView-test',
                 useSubjectBatches=1, useCamBatches=2,
                 batch_size=8,
                 randomize=True)

    trainloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
                                              num_workers=0, pin_memory=False,
                                              collate_fn=utils_data.default_collate_with_string)

    # iterate over batches
    for input, labels in iter(trainloader):
        IPython.embed()

