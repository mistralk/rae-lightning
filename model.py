import torch
from torch import nn, Tensor
from torch.optim import lr_scheduler, Adam
import pytorch_lightning as pl
from utils import *
from typing import List, Optional, Sequence, Tuple, Any, Callable


class RAEModel(pl.LightningModule):
    def __init__(self, in_channels, sequence_length):
        super().__init__()
        self.model = RAE(in_channels=in_channels)
        self.loss_weights = [0.8, 0.1, 0.1]
        self.temporal_weights = [0.011, 0.044, 0.135, 0.325, 0.607, 0.882, 1.0]
        self.sequence_length = sequence_length
    
    def forward(self, I, h):
        x_hat, hidden = self.model.forward(I, h)
        return x_hat, hidden
    
    def training_step(self, batch, batch_idx):
        x = batch['img_sequence']
        target = batch['target_sequence']
        # albedo = batch['albedo_sequence']

        # feed-forward the first frame of each sequence
        denoised, hidden = self.forward(x[0], None)
        sequence_loss = torch.zeros(self.sequence_length)
        sequence_loss[0] = self.loss(denoised, target[0]) * self.temporal_weights[0]

        # feed-forward later frames of each sequence
        for i in range(1, self.sequence_length):
            denoised, hidden = self.forward(x[i], hidden)
            sequence_loss[i] = self.loss(denoised, target[i]) * self.temporal_weights[i]

        loss = sequence_loss.mean()
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.99))

    def loss(self, denoised, target):
        loss_s = torch.functional.F.l1_loss(denoised, target)
        loss = self.loss_weights[0] * loss_s # spatial-only loss
        return loss


class RAE(nn.Module):

    def __init__(self,
                 in_channels,
                 encoder_hidden_dims=[32, 32, 43, 57, 76, 101, 101],
                 decoder_hidden_dims=[76, 76, 57, 57, 43, 43, 32, 32, 128, 64]):
        super(RAE, self).__init__()
        
        """Encoder"""
        self.encoder_conv = []
        self.rcnn = []
        self.subsample = nn.MaxPool2d(2)

        for i, h_dim in enumerate(encoder_hidden_dims):
            self.encoder_conv.append(nn.Sequential(
                                     nn.Conv2d(in_channels, out_channels=h_dim, 
                                               kernel_size=3, padding='same'),
                                     nn.LeakyReLU()))
            if i > 0:
                self.rcnn.append(RCNNBlock(h_dim))
            in_channels = h_dim
        
        self.encoder_conv = nn.ModuleList(self.encoder_conv)
        self.rcnn = nn.ModuleList(self.rcnn)

        """Decoder"""
        self.decoder_conv = []
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        for i, h_dim in enumerate(decoder_hidden_dims):
            if i % 2 == 0:
                in_channels = in_channels * 2

            self.decoder_conv.append(nn.Sequential(
                                     nn.Conv2d(in_channels, out_channels=h_dim, 
                                               kernel_size=3, padding='same'),
                                     nn.LeakyReLU()))
            in_channels = h_dim
        
        self.decoder_conv = nn.ModuleList(self.decoder_conv)
        self.output = nn.Conv2d(in_channels, out_channels=3,
                                kernel_size=3, padding='same')

    def encode(self, I, h: List[Tensor]):
        """
        Args:
            I: the current frame
            h: the encoding stage outputs from the previous frame

        Returns:
            an final encoded image and 
            a list of outputs at each encoding stage for skip connection
        """
        connections = []
        x = I

        for i, conv in enumerate(self.encoder_conv):
            print('Encoding layer_{} input: {}'.format(i, str(x.shape)))
            x = conv(x)
            if i > 0:
                x = self.rcnn[i-1](x, h[i-1] if h != None else x)
                connections.append(x)

                if i < len(self.encoder_conv) - 1:
                    x = self.subsample(x)
            print('Encoding layer_{} output: {}'.format(i, str(x.shape)))
        
        print('-----')
        
        return x, connections

    def decode(self, encoded, connections: List[Tensor]):
        x = encoded
        connections = list(reversed(connections))
        del connections[0]

        for i in range(0, len(self.decoder_conv), 2):
            x = self.upsample(x)
            x = torch.concat((x, connections[i//2]), axis=1)
            x = self.decoder_conv[i](x)
            print('Decoding layer_{} output: {}'.format(i, str(x.shape)))
            if i + 1 < len(self.decoder_conv):
                x = self.decoder_conv[i+1](x)
            print('Decoding layer_{} output: {}'.format(i+1, str(x.shape)))

        x = self.output(x)
        print('-----')
        print('Output shape: {}'.format(str(x.shape)))
        
        return x
    
    def forward(self, I, h):
        """
        Args:
            I: the current frame
            h: the encoding stage outputs from the previous frame

        Returns:
            connections will be used for the encoding stages for the next frame
            as a previous hidden state
        """

        encoded, connections = self.encode(I, h)
        decoded = self.decode(encoded, connections)
        
        return decoded, connections


class RCNNBlock(nn.Module):

    def __init__(self, channels):
        super(RCNNBlock, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels,
                                             kernel_size=3, padding='same'),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.conv2 = nn.Sequential(nn.Conv2d(channels + channels, channels,
                                             kernel_size=3, padding='same'),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.conv3 = nn.Sequential(nn.Conv2d(channels, channels,
                                             kernel_size=3, padding='same'),
                                   nn.LeakyReLU(negative_slope=0.1))
    
    def forward(self, input, hidden_state):
        x = self.conv1(input)
        x = torch.concat((x, hidden_state), axis=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x