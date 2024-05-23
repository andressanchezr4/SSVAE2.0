# SSVAE2.0
### SemiSupervised Variational Autoencoder updated to tensorflow 2.0+

This is the updated code for tensorflow 2.0+ for the model described in the paper "Conditional molecular design with deep generative models". See the [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00263) here.

## Some changes with respect to the original publication
- SSVAE2.0 now supports an arbitrary number of molecule properties.
- The way batches are created is up to date, no need to directly load numpy arrays to the model.
- The stop condition for the training is now stablished as its own class.
- It can easily be run in the GPU by initializing spyder with the variable -> CUDA_VISIBLE_DEVICES=0 spyder




