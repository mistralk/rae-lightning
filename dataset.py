import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
import numpy as np
import pathlib
from utils import *


class RAEDataModule(LightningDataModule):

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def build(self, dataset_dir, aux_features, sequence_length):
        self.sequence_length = sequence_length

        dataset = RAEDataset(dataset_dir, aux_features, sequence_length)
        
        num_train = int(len(dataset) * 0.9)
        num_test = len(dataset) - num_train
        assert num_train + num_test == len(dataset)

        #self.train, self.val = random_split(dataset, [num_train, num_test])
        self.train = dataset

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class RAEDataset(Dataset):

    def __init__(self, root, aux_features, sequence_length):
        self.root = pathlib.Path(root)
        self.aux_features = aux_features

        scene_paths = list(self.root.glob('*'))
        self.scenes = [str(path).split('/')[-1] for path in scene_paths]
        self.frames_per_scene = 2 # would be 1000
        self.num_noisy_image = 5

        self.data = []
        for scene in self.scenes:
            for frame_i in range(self.frames_per_scene - sequence_length + 1):
                frame_name = 'frame-' + str(frame_i).rjust(4, '0')
                path = self.root/scene/frame_name
                for noise_i in range(self.num_noisy_image):
                    self.data.append(path/('noisy-' + str(noise_i) + '.exr'))

        self.sequence_length = sequence_length
    
    # get a sequence [idx, idx + length)
    def __getitem__(self, idx):
        data = {'img_sequence': [],
                'target_sequence': [],
                'albedo_sequence': []}
    
        start_frame = int(self.data[idx].parent.name.split('-')[-1])
        noise_instance = '/' + self.data[idx].name
        
        for i in range(start_frame, start_frame + self.sequence_length):
            path = str(self.data[idx].parent.parent) + '/frame-' + str(i).rjust(4, '0')

            sample_img = []
            sample_target = []
            sample_albedo = [] # for final rendering
            
            for channel_name in ['R', 'G', 'B']:
                sample_img.append(exr_to_numpy(path + noise_instance, channel_name))
                sample_target.append(exr_to_numpy(path + '/target.exr', channel_name))

            for channel_name in self.aux_features:
                sample_img.append(exr_to_numpy(path + '/aux.exr', channel_name))

            for channel_name in ['albedo.R', 'albedo.G', 'albedo.B']:
                sample_albedo.append(exr_to_numpy(path + '/aux.exr', channel_name))

            # TODO: albedo demodulation from input RGB
            sample_img = np.stack(sample_img)
            sample_target = np.stack(sample_target)

            data['img_sequence'].append(sample_img)
            data['target_sequence'].append(sample_target)
            data['albedo_sequence'].append(sample_albedo)

        print_srgb(data['img_sequence'][0])
        return data

    def __len__(self):
        return len(self.data)