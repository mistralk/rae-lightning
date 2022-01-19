import click
import os
from model import *
from dataset import *
from utils import *
from PIL import Image

@click.command()
@click.argument(
    'ckpt',
    type=click.Path(exists=True)
)
@click.argument(
    'img_path',
    type=click.Path(exists=True)
)
def inference(
    ckpt,
    img_path
):
    
    aux_features = ['depth.Z', 'normal.R', 'normal.G', 'normal.B']
    seq = load_sequence(img_path, aux_features)
    device = torch.device('cuda')

    model = RAEModel.load_from_checkpoint(ckpt, num_aux_channels=len(aux_features), sequence_length=seq.shape[0])
    model.eval()
    model.freeze()
    model.to(device)

    with torch.no_grad():
        x = torch.tensor(seq, device=device).unsqueeze(dim=1)
        recons = []

        for i in range(x.shape[0]):
            if i == 0:
                denoised, hidden = model(x[0], None)
            else:
                denoised, hidden = model(x[i], hidden)

            d = denoised[0]
            d = (d - d.min())/(d.max() - d.min())
            d = torch.pow(d, 1/0.2)
            d = torch.pow(d, 1.0/10) * 255
            d = d.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            recons.append(Image.fromarray(d))
            # recons[i].save(f'{i}.png')

    recons[0].save('./out.gif', format='gif', save_all=True, append_images=recons[1:], duration=100, loop=0)

def load_sequence(path, aux_features):
    frame_names = sorted(os.listdir(path))
    channels = ['R', 'G', 'B'] + aux_features

    seq = []

    for frame in frame_names:
        img = exr_to_dict(f'{path}/{frame}/noisy-0.exr', channels)
        target = exr_to_dict(f'{path}/{frame}/target.exr', channels)
        for channel in 'RGB':
            img[channel] = np.power(img[channel], 0.2)

        # normalize Z-buffer
        if 'depth.Z' in target:
            _numer = target['depth.Z'] - target['depth.Z'].min()
            _denom = target['depth.Z'].max() - target['depth.Z'].min()
            if _denom == 0:
                target['depth.Z'] = 0
            else:
                target['depth.Z'] = _numer / _denom
        
        img['depth.Z'] = target['depth.Z']
        img['normal.R'] = target['normal.R']
        img['normal.G'] = target['normal.G']
        img['normal.B'] = target['normal.B']
        
        img = np.stack([img[channel] for channel in channels])
        seq.append(img)
    
    return np.stack(seq)


if __name__ == '__main__':
    inference()