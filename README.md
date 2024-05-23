# SSVAE2.0
### SemiSupervised Variational Autoencoder updated to tensorflow 2.0+

This is the updated code for tensorflow 2.0+ for the model described in the paper "Conditional molecular design with deep generative models". See the [paper here](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00263).

## Some changes with respect to the original publication
- SSVAE2.0 molecule conditional prediction now supports an arbitrary number of molecule properties.
- The way batches are generated is up to date and implemented in the full_preprocessing.py. No need to directly load numpy arrays to the model.
- The stop condition for the training is now defined as its own class.
- The model weights generated after training the model with the original dataset split is included.
- It can easily be run in the GPU by initializing spyder with the enviroment variable -> CUDA_VISIBLE_DEVICES=0 spyder




