import numpy as np
import matplotlib.pyplot as plt
import Imath
import OpenEXR

"""
EXR proceessing code is adapted from 
https://cgcooke.github.io/Blog/computer%20vision/blender/2020/10/30/Training-Data-From-OpenEXR.html
"""


def hdr_normalize(v):
    darkest = v.min()
    lighest = v.max()
    v = v - darkest
    scale = 1.0 / (lighest - darkest)
    return v * scale

def encode_to_SRGB(v):
    return np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055)
    
def correct_gamma(data):
    return np.power(data, (1.0/2.2))

def numpy_to_srgb(data):
    #srgb = encode_to_SRGB(data[0:3])
    srgb = data
    srgb = srgb.transpose((1, 2, 0))
    srgb = correct_gamma(srgb)

    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(srgb)

    print(srgb.shape)

    return srgb

def print_srgb(data):
    data = data.numpy()
    
    srgb = data[0:3] # get RGB from tensor
    srgb = srgb.transpose((1, 2, 0))
    srgb = correct_gamma(srgb)

    fig = plt.figure()
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    plt.imshow(srgb)
    plt.show()

def exr_to_numpy(exr_path, channels):
    img = OpenEXR.InputFile(str(exr_path))
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_str = img.channels(channels, float_type)
    
    out_channels = []
    for channel_str in channels_str:
        out_channels.append(np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1))

    return np.stack(out_channels)

def exr_to_dict(exr_path, channels):
    img = OpenEXR.InputFile(str(exr_path))
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channels_str = img.channels(channels, float_type)
    
    out = {}
    for channel_name, channel_str in zip(channels, channels_str):
        out[channel_name] = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)

    return out