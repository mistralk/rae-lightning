# Recurrent Denoising Autoencoder - PyTorch Lightning

PyTorch Lightning implementation of [Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder](https://research.nvidia.com/publication/interactive-reconstruction-monte-carlo-image-sequences-using-recurrent-denoising)(2017). This repository is not official implementation. Also, some features in the original paper have not implemented yet.

## Dataset

## Prerequisites

- libopenexr-dev library
- openexr python package
    - `pip install openexr` or, `conda install openexr-python`

## Usage

## Results

## Differences from the original paper (to-implement)

- [ ] Albedo demodulation for denoising and re-modulation for final rendering
- [ ] Using view-space shading normals for input G-Buffer : currently the network uses the world-space one.
- [ ] Loss function - gradient-domain loss and temporal loss : currently the network uses spatial loss only.