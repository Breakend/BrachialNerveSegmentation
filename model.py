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
from lasagne.regularization import regularize_layer_params, l2
import lasagne

from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm


def jaccard_distance(rounded_prediction, rounded_truth):
    n = T.cast(T.sum(rounded_prediction * rounded_truth), 'float32')
    a = T.cast(T.sum(rounded_prediction), 'float32')
    b = T.cast(T.sum(rounded_truth), 'float32')
    z = a + b - n
    
    return T.switch(T.eq(a + b, 0.), 0.0, 1.0 - ((n + 10.) / (z + 10.)))

def inverse_dice(rounded_prediction, rounded_truth):
    #rounded_truth = T.iround(rounded_truth)
    #n = T.cast(T.sum(rounded_prediction & rounded_truth), 'float32')
    #a = T.cast(T.sum(rounded_prediction), 'float32')
    #b = T.cast(T.sum(rounded_truth), 'float32')
    a = -2*(T.sum(rounded_prediction*rounded_truth)+10)/(T.sum(rounded_prediction) + T.sum(rounded_truth) + 10)
    #return -2.0 * ( (n + 100.) / (a + b + 100.))
    

def build_cnn_old(patch_size, input_var=None):
    input_layer = lasagne.layers.InputLayer(shape=(None, 1, patch_size[0], patch_size[1]),
                                        input_var=input_var)

    network = batch_norm(lasagne.layers.Conv2DLayer(input_layer, num_filters=32, filter_size=(9,9)))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5)))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5)))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5)))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3)))
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5), pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3), pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3), pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(3,3), pad='full'))
    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)
    network = batch_norm(lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(5,5), pad='full', nonlinearity=lasagne.nonlinearities.sigmoid))

    return network

def build_cnn(patch_size, input_var=None, n=1):
    # create a residual learning building block with two stacked 3x3 convlayers as in paper
    def residual_block(l, increase_dim=False, projection=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            first_stride = (2,2)
            out_num_filters = input_num_filters*2
        else:
            first_stride = (1,1)
            out_num_filters = input_num_filters

        stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        if increase_dim:
            if projection:
                # projection shortcut, as option B in paper
                projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            else:
                # identity shortcut, as option A in paper
                identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        else:
            block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        return block

    # Building the network
    l_in = InputLayer(shape=(None, 1, patch_size[0], patch_size[1]), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    network = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        network = residual_block(network)

    # second stack of residual blocks, output is 32 x 16 x 16
    network = residual_block(network, increase_dim=True)
    for _ in range(1,n):
        network = residual_block(network)

    # third stack of residual blocks, output is 64 x 8 x 8
    network = residual_block(network, increase_dim=True)
    for _ in range(1,n):
        network = residual_block(network)
    
    import pdb; pdb.set_trace()
    # As in http://arxiv.org/pdf/1506.06204v2.pdf
    # 1x1 conv relu layer followed by classification layer
    network = ConvLayer(network, num_filters=512, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)

    network = ConvLayer(network, num_filters=256, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

    network = lasagne.layers.Upscale2DLayer(network, scale_factor=2)

    network = ConvLayer(network, num_filters=1, filter_size=(1,1), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False)

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
    loss += regularize_layer_params(network, l2) * .0001
    # loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    lr = 0.1
    sh_lr = theano.shared(lasagne.utils.floatX(lr))
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=sh_lr)

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

