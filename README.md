# SSVAE2.0
### SemiSupervised Variational Autoencoder updated to Tensorflow 2.0+

This is the updated code for Tensorflow 2.0+ for the model described in the paper "Conditional molecular design with deep generative models". See the [paper here](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00263). As well, the Tensorflow 1.0 code can be found [here](https://github.com/nyu-dl/conditional-molecular-design-ssvae/tree/master)

## Some changes with respect to the original publication
- SSVAE2.0 molecule conditional prediction now supports an arbitrary number of molecule properties.
- The way batches are generated is up to date and implemented in the full_preprocessing.py. No need to directly load numpy arrays to the model.
- The stop condition for the training is now defined as its own class.

## Requirements
- Tensorflow 2.0+
- Scikit-learn
- RDKit
- Pandas/NumPy


