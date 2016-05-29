#!/usr/bin/env python

from __future__ import print_function

import sys
import gc
import os
import time
import yaml
import traceback

import numpy as np
import theano
import theano.tensor as T
from lasagne.layers.normalization import batch_norm
import lasagne

def jaccard_distance(rounded_prediction, rounded_truth):
    n = T.cast(T.sum(rounded_prediction * rounded_truth), 'float32')
    a = T.cast(T.sum(rounded_prediction), 'float32')
    b = T.cast(T.sum(rounded_truth), 'float32')
    z = a + b - n
    
    return T.switch(T.or_(T.eq(a + b, 0.), T.eq(z, 0.)), 0.0, 1.0 - ((n + 10.) / (z + 10.)))

def inverse_dice(rounded_prediction, rounded_truth):
    #rounded_truth = T.iround(rounded_truth)
    #n = T.cast(T.sum(rounded_prediction & rounded_truth), 'float32')
    #a = T.cast(T.sum(rounded_prediction), 'float32')
    #b = T.cast(T.sum(rounded_truth), 'float32')
    a = -2*(T.sum(rounded_prediction*rounded_truth)+10)/(T.sum(rounded_prediction) + T.sum(rounded_truth) + 10)
    #return -2.0 * ( (n + 100.) / (a + b + 100.))
    

def build_cnn(patch_size, input_var=None):
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, patch_size[0], patch_size[1]),
                                        input_var=input_var)

    network = batch_norm(lasagne.layers.Conv2DLayer(input_layer, num_filters=8, filter_size=5))
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=3))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=4))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=4))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=5))
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=5, pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=4, pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=16, filter_size=4, pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=8, filter_size=3, pad='full'))
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=5, pad='full', nonlinearity=lasagne.nonlinearities.sigmoid))

    return network

def compose_prediction_functions(patch_size, scope="default"):
    input_var = T.ftensor4(scope + 'inputs')
    target_var = T.ftensor4(scope + 'targets')
    network = build_cnn(patch_size, input_var)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_prediction = T.iround(test_prediction)
    val_fn = theano.function([input_var], [test_prediction])
    return network, val_fn

def compose_functions(patch_size, scope="default"):
    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4(scope + 'inputs')
    #TODO: change labels to just be 1
    target_var = T.ftensor4(scope + 'targets')
    network = build_cnn(patch_size, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    #prediction = prediction.reshape((1, prediction.shape[0]))
    #loss = inverse_dice(prediction, target_var)
    loss = jaccard_distance(prediction, target_var)
    # loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=.003)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = inverse_dice(test_prediction, target_var)
    test_loss = jaccard_distance(test_prediction, target_var)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])
    return network, train_fn, val_fn

