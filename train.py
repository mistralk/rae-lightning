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
def train(
    data_path,
    seq_length
):
    aux_features = ['depth.Z', 'normal.R', 'normal.G', 'normal.B']

    datamodule = RAEDataModule(data_path, aux_features, seq_length, batch_size=10)
    model = RAEModel(num_aux_channels=len(aux_features), sequence_length=seq_length)

    #torchinfo.summary(model, (16, 3 + len(aux_features), 128, 128))

    logger = TensorBoardLogger('tb_logs', name='rae')
    trainer = Trainer(gpus=1, max_epochs=1, log_every_n_steps=1, logger=logger)
    #trainer.fit(model, datamodule, ckpt_path='tb_logs/rae/version_16/checkpoints/epoch=2-step=158.ckpt')
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