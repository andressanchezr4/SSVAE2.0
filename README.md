# SSVAE2.0
SSVAE is a Generative Model for molecular design that enables the generation of new molecules according to desired physicochemical properties.

### SemiSupervised Variational Autoencoder updated to Tensorflow 2.0+

This is the updated code for Tensorflow 2.0+ for the model described in the paper "Conditional molecular design with deep generative models". See the [paper here](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00263). As well, the Tensorflow 1.0 code can be found [here](https://github.com/nyu-dl/conditional-molecular-design-ssvae/tree/master).

## Some changes with respect to the original publication
- SSVAE2.0 molecule conditional prediction now supports an arbitrary number of molecule properties.
- The way batches are generated is up to date and implemented in the full_preprocessing.py.
- The stop condition for the training is now defined as its own class.

## Requirements
- Tensorflow 2.0+
- Scikit-learn
- RDKit
- Pandas/NumPy


