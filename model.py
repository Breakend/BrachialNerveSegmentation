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

import lasagne


def categorical_crossentropy_logdomain(log_predictions, targets):
    return -T.sum(targets * log_predictions, axis=1)

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def build_cnn(patch_size, input_var=None):
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, patch_size[0], patch_size[1]),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            input_layer, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    #long_network = lasagne.layers.Conv2DLayer(
    #        input_layer, num_filters=128, filter_size=(13, 13),
    #        nonlinearity=lasagne.nonlinearities.rectify)
    #long_network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 1))

    #network = lasagne.layers.ConcatLayer([network, long_network])

    #network = lasagne.layers.Conv2DLayer(
    #        network, num_filters=32, filter_size=(5, 5),
    #        nonlinearity=lasagne.nonlinearities.rectify)
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.25),
            num_units=1,
            nonlinearity=log_softmax)

    return network


def compose_functions(patch_size, scope="default", batch_size=5):
    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4(scope + 'inputs')
    target_var = T.fmatrix(scope + 'targets')
    network = build_cnn(patch_size, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    #prediction = prediction.reshape((1, prediction.shape[0]))
    loss = lasagne.objectives.squared_error(prediction, target_var).mean()
    # loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.

    test_prediction = lasagne.layers.get_output(network)
    test_loss = lasagne.objectives.squared_error(test_prediction, target_var).mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])
    return network, train_fn, val_fn

