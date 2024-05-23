#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 22:07:44 2024

@author: andres
"""

import time
import tensorflow as tf
from keras import layers
from keras.layers import Dense, GRU, Bidirectional, concatenate, TimeDistributed, RepeatVector, Flatten
from tensorflow import keras
import numpy as np
from numpy.random import seed
seed(99)
tf.random.set_seed(1234)

@tf.keras.utils.register_keras_serializable()
class Sampler(layers.Layer):
    def call(self, data):
        z_mean, z_log_var = data
        batch_size1 = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size1, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    def __init__(self, seqlen_x, dim_y, dim_h, dim_x, dim_z, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.seqlen_x = seqlen_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.dense_peek_in = Dense(dim_x, activation='sigmoid')
        self.repeat_vector = RepeatVector(seqlen_x)
        self.bidirectional_gru1 = Bidirectional(GRU(dim_h, return_sequences=True, return_state=False), name='enc_lab_1')
        self.bidirectional_gru2 = Bidirectional(GRU(dim_h, return_sequences=True, return_state=False), name='enc_lab_2')
        self.bidirectional_gru3 = Bidirectional(GRU(dim_h, return_sequences=False, return_state=True), name='enc_lab_3')
        self.dense_state_out = Dense(dim_z)
        self.z_mean_layer = Dense(dim_z, name="z_mean_lab")
        self.z_log_var_layer = Dense(dim_z, name="z_log_var_lab")
        self.sampler = Sampler(name='sampler_enc_lab')

    def call(self, inputs):
        x_L, y_L = inputs
        
        peek_in = self.dense_peek_in(y_L)
        peek = self.repeat_vector(peek_in)
        encoder_concat = tf.concat([x_L, peek], axis=2)
        
        gru_out1 = self.bidirectional_gru1(encoder_concat)
        gru_out2 = self.bidirectional_gru2(gru_out1)
        _, state_enc_f, state_enc_b = self.bidirectional_gru3(gru_out2)
        
        final_state = tf.concat([state_enc_f, state_enc_b], axis=1)
        state_out = self.dense_state_out(final_state)
        
        z_L_mu = self.z_mean_layer(state_out)
        z_L_lsgms = self.z_log_var_layer(state_out)
        z_L_sample = self.sampler([z_L_mu, z_L_lsgms])
        
        return z_L_mu, z_L_lsgms, z_L_sample
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'seqlen_x': self.seqlen_x,
            'dim_y': self.dim_y,
            'dim_h': self.dim_h,
            'dim_x': self.dim_x,
            'dim_z': self.dim_z
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class Predictor(tf.keras.layers.Layer):
    def __init__(self, seqlen_x, dim_y, dim_h, dim_x, **kwargs):
        super(Predictor, self).__init__(**kwargs)
        self.seqlen_x = seqlen_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_x = dim_x

        self.bidirectional_gru1 = Bidirectional(GRU(dim_h, return_sequences=True, return_state=False), name='pred_lab_1')
        self.bidirectional_gru2 = Bidirectional(GRU(dim_h, return_sequences=True, return_state=False), name='pred_lab_2')
        self.bidirectional_gru3 = Bidirectional(GRU(dim_h, return_sequences=False, return_state=True), name='pred_lab_3')
        self.dense_classifier = Dense(dim_y, name='Dense_after_gru_pred')
        self.y_mean_layer = Dense(dim_y, name="y_mean_lab")
        self.y_log_var_layer = Dense(dim_y, name="y_log_var_lab")
        self.sampler = Sampler(name='sampler_pred_lab')

    def call(self, inputs):
        x_L = inputs
        
        gru_out1 = self.bidirectional_gru1(x_L)
        gru_out2 = self.bidirectional_gru2(gru_out1)
        _, state_pred_f, state_pred_b = self.bidirectional_gru3(gru_out2)
        
        final_state_p = tf.concat([state_pred_f, state_pred_b], axis=1)
        classifier_L_out = self.dense_classifier(final_state_p)
        
        y_L_mu = self.y_mean_layer(classifier_L_out)
        y_L_lsgms = self.y_log_var_layer(classifier_L_out)
        y_L_sample = self.sampler([y_L_mu, y_L_lsgms])
        
        return y_L_mu, y_L_lsgms, y_L_sample
    
    def get_config(self):
        config = super(Predictor, self).get_config()
        config.update({
            'seqlen_x': self.seqlen_x,
            'dim_y': self.dim_y,
            'dim_h': self.dim_h,
            'dim_x': self.dim_x
        })
        return config

@tf.keras.utils.register_keras_serializable() 
class Decoder(tf.keras.layers.Layer):
    def __init__(self, seqlen_x, dim_y, dim_h, dim_x, dim_z, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.seqlen_x = seqlen_x
        self.dim_y = dim_y
        self.dim_h = dim_h
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.dense_peek_in = Dense(dim_x, activation='sigmoid')
        self.repeat_vector = RepeatVector(seqlen_x)
        self.gru1 = GRU(dim_h, return_sequences=True, return_state=False, name='dec_lab_1')
        self.gru2 = GRU(dim_h, return_sequences=True, return_state=False, name='dec_lab_2')
        self.gru3 = GRU(dim_h, return_sequences=True, return_state=False, name='dec_lab_3')
        self.time_distributed = TimeDistributed(Dense(dim_x, activation='softmax'), name='x_L_recon')

    def call(self, inputs):
        z_input, y_L, xs_L = inputs
        
        latent_input_plus = tf.concat([z_input, y_L], 1)
        
        peek_in = self.dense_peek_in(latent_input_plus)
        peek = self.repeat_vector(peek_in)
        decoder_concat_final = concatenate([xs_L, peek], axis=2, name='concat_2')
        
        d_gru1 = self.gru1(decoder_concat_final)
        d_gru2 = self.gru2(d_gru1)
        d_gru3 = self.gru3(d_gru2)
        x_L_recon = self.time_distributed(d_gru3)
        
        return x_L_recon
    
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'seqlen_x': self.seqlen_x,
            'dim_y': self.dim_y,
            'dim_h': self.dim_h,
            'dim_x': self.dim_x,
            'dim_z': self.dim_z
        })
        return config

@tf.keras.utils.register_keras_serializable()
class SSVAE(tf.keras.Model):
    def __init__(self, trnX_L, trnX_U, trnY_L, batch_size, 
                 beta = 10000, n_hidden1 = 250, n_hidden2 = 100, **kwargs):
        super(SSVAE, self).__init__(**kwargs)
        
        self.mu_prior = np.mean(trnY_L, 0)
        self.cov_prior = np.cov(trnY_L.T)
        
        self.len_x = trnX_L.shape[0]
        self.len_xu = trnX_U.shape[0]
        
        self.seqlen_x = trnX_L.shape[1]
        self.dim_x = trnX_L.shape[2]
        self.dim_h = n_hidden1
        self.dim_z = n_hidden2
        self.dim_y = trnY_L.shape[1]
        self.batch_size = batch_size
        
        char_set = [' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '#', '(', ')', '[', ']',
                  '+', '=', 'B', 'Br', 'c', 'C', 'Cl', 'F', 'H', 'I', 'N', 'n', 'O', 'o', 'P', 'p',
                  'S', 's', 'Si', 'Sn']
        self.int_to_char = {i: c for i, c in enumerate(char_set)}
        
        self.beta = beta
        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.objL_loss_tracker = keras.metrics.Mean(name="objL_loss")
        self.objU_loss_tracker = keras.metrics.Mean(name="objU_loss")
        self.y_loss_tracker = keras.metrics.Mean(name="y_loss")
        
        self.predictor = Predictor(self.seqlen_x, self.dim_y, self.dim_h, self.dim_x)
        self.encoder = Encoder(self.seqlen_x, self.dim_y, self.dim_h, self.dim_x, self.dim_z)
        self.decoder = Decoder(self.seqlen_x, self.dim_y, self.dim_h, self.dim_x, self.dim_z)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.objL_loss_tracker,
            self.objU_loss_tracker,
            self.y_loss_tracker,
        ]

    def call(self, inputs):
        x, xs, y, xu, xsu = inputs
        z_L_mu, z_L_lsgms, z_L_sample = self.encoder([x, y])
        y_L_mu, y_L_lsgms, y_L_sample = self.predictor(x)
        x_L_recon = self.decoder([z_L_sample, y, xs])
        return x_L_recon

    def train_step(self, data):
        inputs, _ = data
        x, xs, y, xu, xsu = inputs
        
        batch_size_L = self.batch_size*self.len_x/self.len_x+self.len_xu
        batch_size_U = self.batch_size*self.len_xu/self.len_x+self.len_xu
        
        self.tf_mu_prior = tf.constant(self.mu_prior, shape=[1, self.dim_y], dtype=tf.float32)
        self.tf_cov_prior = tf.constant(self.cov_prior, shape=[self.dim_y, self.dim_y], dtype=tf.float32)

        with tf.GradientTape() as tape:
            #LABELED
            z_L_mu, z_L_lsgms, z_L_sample = self.encoder([x, y])
            y_L_mu, y_L_lsgms, y_L_sample = self.predictor(x)
            x_L_recon = self.decoder([z_L_sample, y, xs])
            
            #UNLABELED
            y_U_mu, y_U_lsgms, y_U_sample = self.predictor(xu, training=False)
            z_U_mu, z_U_lsgms, z_U_sample = self.encoder([xu, y_U_sample], training= False)
            x_U_recon = self.decoder([z_U_sample, y_U_sample, xsu], training=False)
            
            z_U2_mu, z_U2_lsgms, z_U2_sample = self.encoder([xu, y_U_mu], training=False)
            #x_DU_recon = self.decoder_U([z_U2_sample, y_U_mu, xsu], trainable = False)
            
            objU = self._obj_U(xu, x_U_recon, y_U_mu, y_U_lsgms, z_U_mu, z_U_lsgms)
            objL = self._obj_L(x, y, x_L_recon, z_L_mu, z_L_lsgms)
            objYpred_MSE = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y, y_L_mu), 1))

            total_loss = (objL * float(batch_size_L) + objU * float(batch_size_U))/float(batch_size_L+batch_size_U) + float(batch_size_L)/float(batch_size_L+batch_size_U) * (self.beta * objYpred_MSE)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.objL_loss_tracker.update_state(objL)
        self.objU_loss_tracker.update_state(objU)
        self.y_loss_tracker.update_state(objYpred_MSE)
        
        return {
            "train_total_loss": self.total_loss_tracker.result(),
            "objL_loss": self.objL_loss_tracker.result(),
            "objU_loss": self.objU_loss_tracker.result(),
            "y_loss": self.y_loss_tracker.result()
            }
    
    def test_step(self, val_data):
        inputs, _ = val_data
        x, xs, y, xu, xsu = inputs
        
        # batch_size_L=int(self.batch_size*self.len_x/self.len_x+self.len_xu)
        # batch_size_U=int(self.batch_size*self.len_xu/self.len_x+self.len_xu)
        
        # LABELED
        z_L_mu, z_L_lsgms, z_L_sample = self.encoder([x, y])
        y_L_mu, y_L_lsgms, y_L_sample = self.predictor(x)
        #x_L_recon = self.decoder([z_L_sample, y, xs])
        x_DL_recon = self.decoder([z_L_mu, y, xs])
        
        # UNLABELED
        y_U_mu, y_U_lsgms, y_U_sample = self.predictor(xu)
        z_U_mu, z_U_lsgms, z_U_sample = self.encoder([xu, y_U_sample])
        #x_U_recon = self.decoder([z_U_sample, y_U_sample, xsu])

        z_U2_mu, z_U2_lsgms, z_U2_sample = self.encoder([xu, y_U_mu])
        x_DU_recon = self.decoder([z_U2_sample, y_U_mu, xsu])

        val_objL = - tf.reduce_mean(- tf.reduce_sum(self.cross_entropy(Flatten()(x), Flatten()(x_DL_recon)), 1))
        val_objU = - tf.reduce_mean(- tf.reduce_sum(self.cross_entropy(Flatten()(xu), Flatten()(x_DU_recon)), 1))
        val_objYpred_MSE = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(y, y_L_mu), 1))

        val_total_loss = val_objYpred_MSE #(val_objL * float(batch_size_L) + val_objU * float(batch_size_U))/float(batch_size_L+batch_size_U) + float(batch_size_L)/float(batch_size_L+batch_size_U) * (self.beta * val_objYpred_MSE)
        
        return {
                "loss": val_total_loss,
                'objL_loss': val_objL,
                'objU_loss': val_objU,
                'y_loss': val_objYpred_MSE
                }
    
    def sampling_conditional(self, yid, ytarget):
    
        def random_cond_normal(yid, ytarget):

            id2=[yid]
            id1=np.setdiff1d(list(range(self.dim_y)),id2)
        
            mu1=self.mu_prior[id1]
            mu2=self.mu_prior[id2]
            
            cov11=self.cov_prior[id1][:,id1]
            cov12=self.cov_prior[id1][:,id2]
            cov22=self.cov_prior[id2][:,id2]
            cov21=self.cov_prior[id2][:,id1]
            
            cond_mu = np.transpose(mu1.T+np.matmul(cov12, np.linalg.inv(cov22)) * (ytarget-mu2))[0]
            cond_cov = cov11 - np.matmul(np.matmul(cov12, np.linalg.inv(cov22)), cov21)
            
            marginal_sampled=np.random.multivariate_normal(cond_mu, cond_cov, 1)
            
            tst=np.zeros(self.dim_y)
            tst[id1]=marginal_sampled
            tst[id2]=ytarget
            
            return np.asarray([tst])

        sample_z=np.random.randn(1, self.dim_z)
        sample_y=random_cond_normal(yid, ytarget) 
          
        sample_smiles=self.beam_search(sample_z, sample_y, k=5)
            
        return sample_smiles

    def sampling_unconditional(self):

        sample_z = np.random.randn(1, self.dim_z)
        sample_y = np.random.multivariate_normal(self.mu_prior, self.cov_prior, 1)

        sample_smiles = self.beam_search(sample_z, sample_y, k=5)

        return sample_smiles

    def beam_search(self, z_input, y_input, k=5):

        def reconstruct(xs_input, z_sample, y_input):
            x_L_recon = self.decoder([z_sample, y_input, xs_input])
            return x_L_recon

        cands = np.asarray([np.zeros((1, self.seqlen_x, self.dim_x), dtype=np.float32)])
        cands_score = np.asarray([100.])

        for i in range(self.seqlen_x-1):

            cands2 = []
            cands2_score = []

            for j, samplevec in enumerate(cands):
                o = reconstruct(samplevec, z_input, y_input)
                sampleidxs = np.argsort(-o[0, i])[:k]

                for sampleidx in sampleidxs:

                    samplevectt = np.copy(samplevec)
                    samplevectt[0, i+1, sampleidx] = 1.

                    cands2.append(samplevectt)
                    cands2_score.append(cands_score[j] * o[0, i, sampleidx])

            cands2_score = np.asarray(cands2_score)
            cands2 = np.asarray(cands2)

            kbestid = np.argsort(-cands2_score)[:k]
            cands = np.copy(cands2[kbestid])
            cands_score = np.copy(cands2_score[kbestid])

            if np.sum([np.argmax(c[0][i+1]) for c in cands]) == 0:
                break

        sampletxt = ''.join([self.int_to_char[np.argmax(t)] for t in cands[0, 0]]).strip()

        return sampletxt

    def _obj_L(self, x, y, x_L_recon, z_L_mu, z_L_lsgms):

        L_log_lik = - tf.reduce_sum(self.cross_entropy(Flatten()(x), Flatten()(x_L_recon)), 1)
        L_log_prior_y=self.noniso_logpdf(y)
        L_KLD_z=self.iso_KLD(z_L_mu, z_L_lsgms)

        objL= -tf.reduce_mean(L_log_lik + L_log_prior_y - L_KLD_z)

        return objL
    
    def _obj_U(self, xu, x_U_recon, y_U_mu, y_U_lsgms, z_U_mu, z_U_lsgms):

        U_log_lik = - tf.reduce_sum(self.cross_entropy(Flatten()(xu), Flatten()(x_U_recon)), 1)
        U_KLD_y = self.noniso_KLD(y_U_mu, y_U_lsgms)
        U_KLD_z = self.iso_KLD(z_U_mu, z_U_lsgms)

        objU = -tf.reduce_mean(U_log_lik - U_KLD_y - U_KLD_z)
        
        return objU

    def cross_entropy(self, x, y, const=1e-10):
        return - (x*tf.math.log(tf.clip_by_value(y, const, 1.0))+(1.0-x)*tf.math.log(tf.clip_by_value(1.0-y, const, 1.0)))

    def iso_KLD(self, mu, log_sigma_sq):
        return tf.reduce_sum(-0.5 * (1.0 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq)), 1)

    def noniso_logpdf(self, x):  # la x se corresponde a la y
        return - 0.5 * (float(self.cov_prior.shape[0]) * np.log(2.*np.pi) + np.log(np.linalg.det(self.cov_prior))
                        + tf.reduce_sum(tf.multiply(tf.matmul(tf.subtract(x, self.tf_mu_prior), tf.linalg.inv(self.tf_cov_prior)), tf.subtract(x, self.tf_mu_prior)), 1))

    def noniso_KLD(self, mu, log_sigma_sq):
        return 0.5 * (tf.linalg.trace(tf.scan(lambda a, x: tf.matmul(tf.linalg.inv(self.tf_cov_prior), x), tf.linalg.diag(tf.exp(log_sigma_sq))))
                      + tf.reduce_sum(tf.multiply(tf.matmul(tf.subtract(self.tf_mu_prior, mu),
                                      tf.linalg.inv(self.tf_cov_prior)), tf.subtract(self.tf_mu_prior, mu)), 1)
                      - float(self.cov_prior.shape[0]) + np.log(np.linalg.det(self.cov_prior)) - tf.reduce_sum(log_sigma_sq, 1))
    
    def get_config(self):
        config = super(SSVAE, self).get_config()
        config.update({
            'batch_size': self.batch_size,
            'beta': self.beta,
            'n_hidden1': self.n_hidden1,
            'n_hidden2': self.n_hidden2,
            
        })
        return config

class EarlyStoppingCustomized(tf.keras.callbacks.Callback):

  def __init__(self):
    super(EarlyStoppingCustomized, self).__init__()
    self.val_log = np.zeros(1000)   

  def on_train_begin(self, logs=None):
    self.stopped_epoch = 0
    self.start_training = time.time()

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get('val_loss')
    self.val_log[epoch] = current
    if epoch > 20 and np.min(self.val_log[0:epoch-10]) * 0.99 < np.min(self.val_log[epoch-10:epoch+1]):
        self.stopped_epoch = epoch
        self.model.stop_training = True
        print('---termination condition is met')

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('Epoch %05d: Detencion anticipada' % (self.stopped_epoch + 1))
      self.end_training = time.time()

