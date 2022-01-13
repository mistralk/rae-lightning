import numpy as np
import matplotlib.pyplot as plt
import Imath
import OpenEXR

"""
EXR proceessing code is from 
https://cgcooke.github.io/Blog/computer%20vision/blender/2020/10/30/Training-Data-From-OpenEXR.html
"""

def encode_to_SRGB(v):
    return(np.where(v<=0.0031308,v * 12.92, 1.055*(v**(1.0/2.4)) - 0.055))

def correct_gamma(data):
    return data**(1.0/2.2)

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


def exr_to_numpy(exr_path, channel_name):
    img = OpenEXR.InputFile(str(exr_path))
    dw = img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    float_type = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_str = img.channel(channel_name, float_type)
    channel = np.frombuffer(channel_str, dtype=np.float32).reshape(size[1], -1)

    return channel