# Recurrent Denoising Autoencoder - PyTorch Lightning

A PyTorch Lightning implementation of [Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder](https://research.nvidia.com/publication/interactive-reconstruction-monte-carlo-image-sequences-using-recurrent-denoising)(2017), for study purposes. This repository is not official implementation. Also, some features in the original paper have not implemented.



## Prerequisites

- libopenexr-dev library
- openexr python package
    - You can install by `pip install openexr`, or, `conda install openexr-python` if you are using Anaconda.

## Usage

### Training

```shell
python train.py PATH_TO_TRAIN_SET_ROOT
```

## Dataset structure

- For training, you have to prepare a dataset generated via Monte Carlo path tracing.
- Each input EXR file should contain 7 channels: R/G/B, depth, world-space shading normal x/y/z
- However, you can change input buffer definition(e.g., same as the original paper's description) by modifying the script.

```shell
Dataset root/
├─ Scene A/
│  ├─ frame-0000/
│  │  ├─ target.exr (High-SPP target image)
│  │  ├─ noisy-0.exr (1-SPP noisy image by a different random seed)
│  │  ├─ noisy-1.exr
│  │  ├─ noisy-2.exr
│  │  ├─ noisy-3.exr
│  │  └─ noisy-4.exr
│  │  
│  ├─ frame-0001/
│  ├─ ...
│  └─ frame-####/
│
├─ Scene B/
└─ ...
```


## Results

## Differences from the original paper

- [ ] Using noise-free G-Buffer by rasterization
- [ ] Using view-space shading normals for G-Buffer : currently the network uses the world-space one.
- [ ] Albedo demodulation for denoising and re-modulation for final rendering

## References

- Chakravarty R Alla Chaitanya, Anton S Kaplanyan, Christoph Schied, Marco Salvi, Aaron Lefohn, Derek Nowrouzezahrai, and Timo Aila. 2017. "Interactive reconstruction of Monte Carlo image sequences using a recurrent denoising autoencoder". ACM Transactions on Graphics (TOG) 36, 4 (2017), 1–12.