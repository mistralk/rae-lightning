import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import pathlib
from typing import Optional
from utils import *


class RAEDataModule(LightningDataModule):

    def __init__(self, data_dir, aux_features, seq_length, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.aux_features = aux_features
        self.seq_length = seq_length
        self.batch_size = batch_size

    def setup(self, stage: Optional[str]=None):
        if stage in (None, 'fit'):
            dataset = RAEDataset(self.data_dir, self.aux_features, self.seq_length)
            num_train = int(len(dataset) * 0.9)
            num_valid = len(dataset) - num_train
            self.train, self.val = random_split(dataset, [num_train, num_valid])

        if stage in (None, 'test'):
            self.test_dir = self.data_dir
            self.test = RAEDataset(self.test_dir, self.aux_features, self.seq_length)
        
        if stage in (None, 'predict'):
            self.test = RAEDataset(self.data_dir, self.aux_features, self.seq_length)

        print('Dataset setup completes.')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=24)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=24)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=24)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=24)


class RAEDataset(Dataset):

    def __init__(self, root, aux_features, sequence_length):
        self.root = pathlib.Path(root)
        self.aux_features = aux_features

        scene_paths = list(self.root.glob('*'))
        self.scenes = [str(path).split('/')[-1] for path in scene_paths]
        self.frames_per_scene = 200
        self.num_noisy_image = 5
        self.sequence_length = sequence_length
        self.channels = ['R', 'G', 'B'] + self.aux_features

        self.data = []
        for scene in self.scenes:
            for frame_i in range(self.frames_per_scene - sequence_length + 1):
                frame_name = 'frame-' + str(frame_i).rjust(4, '0')
                path = self.root/scene/frame_name
                self.data.append(path)
    
    # get a sequence [idx, idx + length)
    def __getitem__(self, idx):
        data = {'img_sequence': [],
                'target_sequence': []
                }

        #albedos = ['albedo.R', 'albedo.G', 'albedo.B']
    
        start_frame = int(self.data[idx].name.split('-')[-1])
        noise_frame_k = np.random.randint(0, self.num_noisy_image)
        
        for i in range(start_frame, start_frame + self.sequence_length):
            path = f'{self.data[idx].parent}/frame-{str(i).rjust(4, "0")}'

            sample_img = exr_to_dict(f'{path}/noisy-{noise_frame_k}.exr', self.channels)
        
            if 'depth.Z' in sample_img:
                _numer = sample_img['depth.Z'] - sample_img['depth.Z'].min()
                _denom = sample_img['depth.Z'].max() - sample_img['depth.Z'].min()
                if _denom == 0:
                    sample_img['depth.Z'] = 0
                else:
                    sample_img['depth.Z'] = _numer / _denom
            
            #img_albedo = np.stack([sample_img[channel] for channel in albedos])
            #sample_img['R'] = sample_img['R'] / (img_albedo[0] + 0.00316)
            #sample_img['G'] = sample_img['G'] / (img_albedo[1] + 0.00316)
            #sample_img['B'] = sample_img['B'] / (img_albedo[2] + 0.00316)

            sample_target = exr_to_dict(f'{path}/target.exr', self.channels)

            for channel in 'RGB':
                sample_img[channel] = np.power(sample_img[channel], 0.2)
                sample_target[channel] = np.power(sample_target[channel], 0.2)

            
            """ for testing noise-free g-buffer """
            """
            if 'depth.Z' in sample_target:
                _numer = sample_target['depth.Z'] - sample_target['depth.Z'].min()
                _denom = sample_target['depth.Z'].max() - sample_target['depth.Z'].min()
                if _denom == 0:
                    sample_target['depth.Z'] = 0
                else:
                    sample_target['depth.Z'] = _numer / _denom
            sample_img['depth.Z'] = sample_target['depth.Z']
            sample_img['normal.R'] = sample_target['normal.R']
            sample_img['normal.G'] = sample_target['normal.G']
            sample_img['normal.B'] = sample_target['normal.B']
            """
            #target_albedo = np.stack([sample_target[channel] for channel in albedos])
            #sample_target['R'] = sample_target['R'] / (target_albedo[0] + 0.00316)
            #sample_target['G'] = sample_target['G'] / (target_albedo[1] + 0.00316)
            #sample_target['B'] = sample_target['B'] / (target_albedo[2] + 0.00316)

            sample_img = np.stack([sample_img[channel] for channel in self.channels])
            sample_target = np.stack([sample_target[channel] for channel in 'RGB'])
            
            sample_img = torch.from_numpy(sample_img)
            sample_target = torch.from_numpy(sample_target)

            i, j, h, w = transforms.RandomCrop.get_params(sample_img, (128, 128))
            sample_img = TF.crop(sample_img, i, j, h, w)
            sample_target = TF.crop(sample_target, i, j, h, w)


            data['img_sequence'].append(sample_img)
            data['target_sequence'].append(sample_target)
            #data['img_albedo'].append(img_albedo)

        return data

    def __len__(self):
        return len(self.data)