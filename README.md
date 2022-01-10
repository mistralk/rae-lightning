# Recurrent Denoising Autoencoder - PyTorch Lightning

A PyTorch Lightning implementation of [Interactive Reconstruction of Monte Carlo Image Sequences using a Recurrent Denoising Autoencoder](https://research.nvidia.com/publication/interactive-reconstruction-monte-carlo-image-sequences-using-recurrent-denoising)(2017), for study purposes. This repository is not official implementation. Also, some features in the original paper have not implemented.

## Dataset

## Prerequisites

- libopenexr-dev library
- openexr python package
    - You can install by `pip install openexr`, or, `conda install openexr-python` if you are using Anaconda.

## Usage

## Results

## Differences from the original paper (to-implement)

- [ ] Albedo demodulation for denoising and re-modulation for final rendering
- [ ] Using view-space shading normals for input G-Buffer : currently the network uses the world-space one.
- [ ] Loss function - gradient-domain loss and temporal loss : currently the network uses spatial loss only.

## References

- Chakravarty R Alla Chaitanya, Anton S Kaplanyan, Christoph Schied, Marco Salvi, Aaron Lefohn, Derek Nowrouzezahrai, and Timo Aila. 2017. "Interactive reconstruction of Monte Carlo image sequences using a recurrent denoising autoencoder". ACM Transactions on Graphics (TOG) 36, 4 (2017), 1–12.