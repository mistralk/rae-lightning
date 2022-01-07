from typing_extensions import Required
import click
from model import *
from dataset import *
from utils import *

from pytorch_lightning import Trainer


@click.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True)
)
@click.option(
    '--seq_length',
    type=int,
    default=2,
    help='Length of consecutive frames to be used during training'
)
def train(
    data_path,
    seq_length
):
    aux_features = ['depth.Z', 'normal.R', 'normal.G', 'normal.B']

    datamodule = RAEDataModule()
    datamodule.build(data_path, aux_features, seq_length)

    model = RAEModel(num_aux_channels=len(aux_features), sequence_length=seq_length)

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model, datamodule)

    print('Training completes!')


if __name__ == '__main__':
    train()