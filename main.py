import click
from model import *
from dataset import *
from utils import *

from pytorch_lightning import Trainer


@click.command()
@click.option('--sequence_length',
              help='Length of consecutive frames to provide enouth temporal context during training')
def cli(*args, **kwargs):
    train(args['sequence_length'])


def train(sequence_length):
    datamodule = RAEDataModule()
    datamodule.build(sequence_length)
    model = RAEModel(in_channels=7, sequence_length=sequence_length)

    trainer = Trainer(gpus=1, max_epochs=1)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    cli()

