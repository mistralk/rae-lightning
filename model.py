import torch
from torch import nn, Tensor
from torch.optim import lr_scheduler, Adam
from torch.nn import functional as F
import pytorch_lightning as pl
import torchvision
from utils import *
from typing import List, Optional, Sequence, Tuple, Any, Callable
from piq import ssim


class RAEModel(pl.LightningModule):

    def __init__(self, num_aux_channels, sequence_length):
        super().__init__()
        in_channels = 3 + num_aux_channels # RGB + aux
        
        #self.model = RAE(in_channels, [32,32,43,57,57], [43,43,32,32,64])
        #self.model = RAE(in_channels, [32,32,43,57,76,101,135,135], [101,101,76,76,57,57,43,43,32,32,128,64])
        self.model = RAE(in_channels)
        self.loss_weights = [0.8, 0.1, 0.1]
        self.temporal_weights = [0.011, 0.044, 0.135, 0.325, 0.607, 0.882, 1.0]
        self.sequence_length = sequence_length

        x = np.linspace(-4, 4, 9)
        y = np.linspace(-4, 4, 9)
        x_grid, y_grid = np.meshgrid(x, y)
        self.LoG_filter = torch.tensor(self._LoG(x_grid, y_grid, 1.5),
                                       dtype=torch.float32,
                                       requires_grad=False).repeat(3, 3, 1, 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.99))
        l = lambda epoch: 1/np.sqrt(epoch) if epoch != 0 else 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[l])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }

    def loss(self, denoised, target):
        """
        Spatial loss
        """
        loss_s = F.l1_loss(denoised, target, reduction='none')

        """
        Gradient-domain loss
        """
        LoG_filter = self.LoG_filter.to(self.device)
        denoised_LoG = []
        target_LoG = []
        for i in range(self.sequence_length):
            denoised_LoG.append(F.conv2d(denoised[i], LoG_filter, padding='same'))
            target_LoG.append(F.conv2d(target[i], LoG_filter, padding='same'))
        denoised_LoG = torch.stack(denoised_LoG)
        target_LoG = torch.stack(target_LoG)
        loss_g = F.l1_loss(denoised_LoG, target_LoG, reduction='none')

        """
        Temporal loss
        """
        denoised_dt = []
        target_dt = []
        denoised_dt.append(denoised[0] - denoised[0])
        target_dt.append(target[0] - target[0])
        for i in range(1, self.sequence_length):
            denoised_dt.append(denoised[i] - denoised[i-1])
            target_dt.append(target[i] - target[i-1])
        denoised_dt = torch.stack(denoised_dt)
        target_dt = torch.stack(target_dt)
        loss_t = F.l1_loss(denoised_dt, target_dt, reduction='none')

        total_loss = (self.loss_weights[0] * loss_s) \
                + (self.loss_weights[1] * loss_g) \
                + (self.loss_weights[2] * loss_t)

        total_loss = torch.mean(total_loss, dim=[1,2,3,4])

        loss = total_loss * torch.tensor(self.temporal_weights, device=self.device)
        return loss.mean()
    
    def forward(self, I, h=None):
        x_hat, hidden = self.model.forward(I, h)
        return x_hat, hidden
    
    def training_step(self, batch, batch_idx):
        x = batch['img_sequence']
        target = batch['target_sequence']
        recons = []

        # feed-forward the first frame of each sequence
        with torch.no_grad():
            denoised, hidden = self.forward(x[0], None)
        recons.append(denoised)

        # later frames of each sequence
        for i in range(1, self.sequence_length):
            denoised, hidden = self.forward(x[i], hidden)
            recons.append(denoised)

        recons = torch.stack(recons)
        target = torch.stack(target)

        sch = self.lr_schedulers()
        if self.trainer.is_last_batch and self.trainer.current_epoch > 10:
            sch.step()

        return self.loss(recons, target)
    
    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch['img_sequence']
        target = batch['target_sequence']
        recons = []

        denoised, hidden = self.forward(x[0], None)
        recons.append(denoised)

        for i in range(1, self.sequence_length):
            denoised, hidden = self.forward(x[i], hidden)
            recons.append(denoised)
        
        recons = torch.stack(recons)
        target = torch.stack(target)
        loss = self.loss(recons, target)
        self.log(f'{prefix}_loss', loss)

        #ssim_loss = ssim(torch.tensor(hdr_normalize(recons[0].cpu())), 
        #                 torch.tensor(hdr_normalize(target[0].cpu())))
        #self.log(f'{prefix}_SSIM', ssim_loss)
        
        grid_recon = torchvision.utils.make_grid(denoised)
        grid_input = torchvision.utils.make_grid(x[-1][:,:3,:])
        grid_target = torchvision.utils.make_grid(target[-1])
        self.logger.experiment.add_image('input', grid_input)
        self.logger.experiment.add_image('reconstructed', grid_recon)
        self.logger.experiment.add_image('target', grid_target)

    def _LoG(self, x, y, sigma):
        return -1.0 / (np.pi * sigma**4) \
                * (1.0 - (x**2 + y**2) / (2 * sigma**2)) \
                * np.e**(-(x**2+y**2) / (2 * sigma**2))


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
            x = conv(x)
            if i > 0:
                x = self.rcnn[i-1](x, h[i-1] if h != None else None)
                connections.append(x)

                if i < len(self.encoder_conv) - 1:
                    x = self.subsample(x)
        
        return x, connections

    def decode(self, encoded, connections: List[Tensor]):
        x = encoded
        connections = list(reversed(connections))
        del connections[0]

        for i in range(0, len(self.decoder_conv), 2):
            x = self.upsample(x)
            x = torch.concat((x, connections[i//2]), axis=1)
            x = self.decoder_conv[i](x)
            if i + 1 < len(self.decoder_conv):
                x = self.decoder_conv[i+1](x)

        x = self.output(x)
        
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
        if hidden_state is None:
            hidden_state = torch.zeros_like(input, requires_grad=False)
        x = torch.concat((x, hidden_state), dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x