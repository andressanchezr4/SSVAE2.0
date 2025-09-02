#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2024

@author: andressanchezr4
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
from tensorflow.keras.utils import Sequence
import pandas as pd
from rdkit.Chem import Descriptors

seed(99)
tf.random.set_seed(1234)

tf.executing_eagerly()
physical_devices = tf.config.list_physical_devices('GPU')

char_set = [' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '#', '(', ')', '[', ']',
          '+', '=', 'B', 'Br', 'c', 'C', 'Cl', 'F', 'H', 'I', 'N', 'n', 'O', 'o', 'P', 'p',
          'S', 's', 'Si', 'Sn']
int_to_char = {i: c for i, c in enumerate(char_set)}
char_to_int = dict((c, i) for i, c in enumerate(char_set))

class DataPrepare(object):
    def __init__(self, data_df, train_perc = 0.9, val_perc = 0.05, test_perc = 0.05, unsupervised_perc = 0.5):
        self.df = data_df
        self.train_perc = int(train_perc*self.df.shape[0])
        self.val_perc = val_perc
        self.test_perc = int(test_perc*self.df.shape[0])
        self.unsup_perc = unsupervised_perc
        
    def vectorize(self, list_input, char_set):
        one_hot = np.zeros(
            (list_input.shape[0], list_input.shape[1]+4, len(char_set)), dtype=np.int32)

        for si, ss in enumerate(list_input):
            for cj, cc in enumerate(ss):
                one_hot[si, cj+1, cc] = 1

            one_hot[si, -1, 0] = 1
            one_hot[si, -2, 0] = 1
            one_hot[si, -3, 0] = 1

        return one_hot[:, 0:-1, :], one_hot[:, 1:, :]


    def smiles_to_seq(self, smiles, char_set):
        list_seq = []
        for s in smiles:
            seq = []
            j = 0
            while j < len(s):
                if j < len(s)-1 and s[j:j+2] in char_set:
                    seq.append(char_to_int[s[j:j+2]])
                    j = j+2

                elif s[j] in char_set:
                    seq.append(char_to_int[s[j]])
                    j = j+1

            list_seq.append(seq)

        list_seq = keras.preprocessing.sequence.pad_sequences(list_seq, padding='post')

        return list_seq        
    
    def Smiles2Tensor(self):
        try:
            smiles = self.df['SMILES'].tolist()
        except:
            print('Make sure the df has "SMILES" as first column')
            print('2: Mol_weight, 3: Logp, 4: TPSA')
        
        list_seq = self.smiles_to_seq(smiles, char_set)
        
        print('We are almost there...')
        Xs, X = self.vectorize(list_seq, char_set)
        Y = self.df[self.df.columns[1:]] # [['MOLWEIGHT', 'LOGP', 'TPSA']]

        X = X[:self.train_perc]
        Xs = Xs[:self.train_perc]
        Y = Y[:self.train_perc]
        
        nL = int(len(Y)*self.unsup_perc)
        nU = len(Y)-nL
        nL_trn = int(nL*(1-self.val_perc))
        nL_val = nL-nL_trn
        nU_trn = int(nU*(1-self.val_perc))
        
        trnX_L = X[:nL_trn]
        trnXs_L = Xs[:nL_trn]
        trnY_L = Y[:nL_trn]

        valX_L = X[nL_trn:nL_trn+nL_val]
        valXs_L = Xs[nL_trn:nL_trn+nL_val]
        valY_L = Y[nL_trn:nL_trn+nL_val]
        
        trnX_U = X[nL_trn+nL_val:nL_trn+nL_val+nU_trn]
        trnXs_U = Xs[nL_trn+nL_val:nL_trn+nL_val+nU_trn]

        valX_U = X[nL_trn+nL_val+nU_trn:]
        valXs_U = Xs[nL_trn+nL_val+nU_trn:]

        self.scaler_Y = StandardScaler()
        self.scaler_Y.fit(Y)
        trnY_L = self.scaler_Y.transform(trnY_L)
        valY_L = self.scaler_Y.transform(valY_L)
        
        print('Done!')
        
        trnY_L = trnY_L.astype('float32')
        trnX_L = trnX_L.astype('float32')
        trnXs_L = trnXs_L.astype('float32')
        trnX_U = trnX_L.astype('float32')
        trnXs_U = trnXs_L.astype('float32')

        valY_L = valY_L.astype('float32')
        valX_L = valX_L.astype('float32')
        valXs_L = valXs_L.astype('float32')
        valX_U = valX_L.astype('float32')
        valXs_U = valXs_L.astype('float32')
        
        tstX = X[-self.test_perc:].astype('float32')
        tstXs = Xs[-self.test_perc:].astype('float32')
        tstY = Y[-self.test_perc:].astype('float32')
        
        return (trnX_L, trnXs_L, trnX_U, trnXs_U, trnY_L), \
               (valX_L, valXs_L, valX_U, valXs_U, valY_L), \
               (tstX, tstXs, tstY)
               
class BatchGenerator(Sequence):
    def __init__(self, x_set, xs_set, xu_set, xsu_set, y_set, batch_size):
        self.x = x_set
        self.xs = xs_set
        self.xu = xu_set
        self.xsu = xsu_set
        self.y = y_set
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.x))

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x_L = self.x[batch_indexes]
        batch_xs_L = self.xs[batch_indexes]
        batch_x_U = self.xu[batch_indexes]
        batch_xs_U = self.xsu[batch_indexes]
        batch_y_L = self.y[batch_indexes]

        inputs = [batch_x_L, batch_xs_L, batch_y_L, batch_x_U, batch_xs_U]

        return inputs, "outputs_dummy"

def smi_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)

    descriptors = {
        'MolWt': Descriptors.MolWt,
        'TPSA': Descriptors.TPSA,
        'LogP': Descriptors.MolLogP,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'NumHDonors': Descriptors.NumHDonors
    }

    results = {desc: func(mol) for desc, func in descriptors.items()}

    return results
          
