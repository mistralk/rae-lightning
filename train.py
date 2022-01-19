from typing_extensions import Required
import click
from model import *
from dataset import *
from utils import *

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# import torchinfo

@click.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True)
)
@click.option(
    '--seq_length',
    type=int,
    default=7,
    help='Length of consecutive frames to be used during training'
)
@click.option(
    '--max_epochs',
    type=int,
    default=1,
    help='Max epochs for training'
)
@click.option(
    '--batch_size',
    type=int,
    default=10,
    help='Training batch size'
)
@click.option(
    '--num_workers',
    type=int,
    default=1,
    help='Number of multiprocess workers for dataloader'
)
@click.option(
    '--ckpt',
    type=click.Path(),
    help='Checkpoint path(.ckpt) for resuming training'
)
def train(
    data_path,
    seq_length,
    max_epochs,
    batch_size,
    num_workers,
    ckpt
):
    aux_features = ['depth.Z', 'normal.R', 'normal.G', 'normal.B']

    datamodule = RAEDataModule(data_path, aux_features, seq_length, batch_size=batch_size, num_workers=num_workers)
    model = RAEModel(num_aux_channels=len(aux_features), sequence_length=seq_length)

    #torchinfo.summary(model, (16, 3 + len(aux_features), 128, 128))

    logger = TensorBoardLogger('tb_logs', name='rae')
    trainer = Trainer(gpus=1, max_epochs=max_epochs, log_every_n_steps=1, logger=logger)

    if ckpt != None:
        trainer.fit(model, datamodule, ckpt_path=ckpt)
    else:
        trainer.fit(model, datamodule)
        
    print('Training completes!')

    trainer.test(ckpt_path='best', datamodule=datamodule)


def visualize_reconstruction(model, sequence):
    model.eval()
    with torch.no_grad():
        denoised = model(sequence)
    print_srgb(denoised)


if __name__ == '__main__':
    train()