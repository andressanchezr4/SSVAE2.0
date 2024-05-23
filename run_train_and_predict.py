#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:20:26 2024

@author: andres
"""

import pandas as pd
import numpy as np
from tensorflow import keras
import os

from full_preprocessing import DataPrepare, BatchGenerator
from SSVAE import SSVAE, EarlyStoppingCustomized

####################
### DATA LOADING ###
####################

# Columns --> 1st: SMILES, 2nd and on: Molecular Properties
data_uri='./data/ZINC_310k.csv'
df = pd.read_csv(data_uri)

# Preparing data
dataset_preparator = DataPrepare(df)

trn_tensors, val_tensors, tst_tensors = dataset_preparator.Smiles2Tensor()

trnX_L, trnX_L, trnXs_L, trnX_U, trnXs_U, trnY_L = trn_tensors
valX_L, valX_L, valXs_L, valX_U, valXs_U, valY_L = val_tensors
tstX, tstXs, tstY = tst_tensors

# Batch generation
batch_size = 128
train_gen = BatchGenerator(trnX_L, trnXs_L, trnX_U, trnXs_U, trnY_L, batch_size)
test_gen = BatchGenerator(valX_L, valXs_L, valX_U, valXs_U, valY_L, batch_size)

#################################
### MODEL INSTANCE & TRAINING ###
#################################

# Create an instance of the model
epochs = 100
my_ssvae = SSVAE(trnX_L, trnX_U, trnY_L, batch_size)
my_callback = EarlyStoppingCustomized()
optimizer = keras.optimizers.Adam(learning_rate = 0.001)
my_ssvae.compile(optimizer=optimizer)

# Train the model
vae_history = my_ssvae.fit(train_gen, callbacks=my_callback,
          epochs=epochs, validation_data=test_gen, shuffle = True)

print(f'It took {((my_callback.end_training - my_callback.start_training)/60)/60} to train')

# Save model Weigths
# my_ssvae.save_weights('./ssvae_weights_310k.cpkt')

###################
### PREDICTIONS ###
###################

# To load the weights we need a new instance of the SSVAE model 
new_model = SSVAE(trnX_L, trnX_U, trnY_L, batch_size)
new_model.load_weights('./ssvae_weights_310k.cpkt')

# unconditional generation
u_beam_search_list=[]
for t in range(10000):
    smi=my_ssvae.sampling_unconditional()
    u_beam_search_list.append(smi)
    print(f'Unconditional generation, molecule {t}: {smi}')

## conditional generation (e.g. MolWt=250)
yid = 0 # the index of the property you want to condition
ytarget = 250. # the value of such property you want your molecules to have
ytarget_transform = (ytarget-dataset_preparator.scaler_Y.mean_[yid])/np.sqrt(dataset_preparator.scaler_Y.var_[yid])
    
c_beam_search_list=[]
for t in range(10000):
    smi = my_ssvae.sampling_conditional(yid, ytarget_transform)
    c_beam_search_list.append(smi)
    print(f'Conditional generation, molecule {t}: {smi}')
