from __future__ import print_function

import cv2
import time
import numpy as np

from data import load_train_data, load_test_data, random_crops

img_rows = 64
img_cols = 80

smooth = 1.
import lasagne
import theano
from tqdm import *
from itertools import izip
import theano.tensor as T
from lasagne.layers import Conv2DLayer as ConvLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import PadLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import Deconv2DLayer
from lasagne.layers import Upscale2DLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import batch_norm

def jaccard_distance(rounded_prediction, rounded_truth):
    n = T.cast(T.sum(rounded_prediction * rounded_truth), 'float32')
    a = T.cast(T.sum(rounded_prediction), 'float32')
    b = T.cast(T.sum(rounded_truth), 'float32')
    z = a + b - n
    
    return T.switch(T.eq(a + b, 0.), 0.0, 1.0 - ((n + 1.) / (z + 1.)))


def build_cnn(input_var=None, n=5):
    
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
    l_in = InputLayer(shape=(None, 1, img_rows, img_cols), input_var=input_var)

    # first layer, output is 16 x 32 x 32
    l = batch_norm(ConvLayer(l_in, num_filters=16, filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    # first stack of residual blocks, output is 16 x 32 x 32
    for _ in range(n):
        l = residual_block(l)

    # second stack of residual blocks, output is 32 x 16 x 16
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)

    # third stack of residual blocks, output is 64 x 8 x 8
    l = residual_block(l, increase_dim=True)
    for _ in range(1,n):
        l = residual_block(l)
    
    network = ConvLayer(l, num_filters=1, filter_size=(1,1), stride=(1,1))
    network = Upscale2DLayer(network, 4)
#    network = Deconv2DLayer(network, num_filters=1, filter_size=(30, 42), stride=2) 
    return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]


def dice_coef(y_true, y_pred):
    y_true_f = T.flatten(y_true)
    y_pred_f = T.flatten(y_pred)
    return (2. * T.dot(y_true_f, T.transpose(y_pred_f)) + smooth) / (T.sum(y_true_f) + T.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def preprocess(imgs, imgs_mask_train = None, number_augs_per_im = 2):
    # TODO: this logic sucks
    # TODO: also rotational invariances?
    if imgs_mask_train is None:
        number_augs_per_im = 0

    imgs_p = np.ndarray((imgs.shape[0]*(number_augs_per_im+1), imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    if imgs_mask_train is not None:
        imgs_masks_p = np.ndarray((imgs.shape[0]*(number_augs_per_im+1), imgs.shape[1], img_rows, img_cols), dtype=np.uint8)

    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
        if imgs_mask_train is not None:
            imgs_masks_p[i, 0] = cv2.resize(imgs_mask_train[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    if imgs_mask_train is None:
        return imgs_p

    for j in range(number_augs_per_im):
        au_img, au_msk = random_crops(imgs, imgs_mask_train, (int(imgs.shape[2] * .8), int(imgs.shape[3]*.8)))
        for i in range(imgs.shape[0]):
            imgs_p[i*(j+2), 0] = cv2.resize(au_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
            imgs_masks_p[i*(j+2), 0] = cv2.resize(au_msk[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    return imgs_p, imgs_masks_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    num_epochs = 500
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train, imgs_mask_train = preprocess(imgs_train, imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    train_indices = np.arange(int(imgs_train.shape[0]*.85))
    np.random.shuffle(train_indices)
    X_train = imgs_train[train_indices]
    Y_train = imgs_mask_train[train_indices]
    Y_test = np.delete(imgs_mask_train, train_indices, 0)
    X_test = np.delete(imgs_train, train_indices, 0)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, 5)
    model = None
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    
    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = jaccard_distance(prediction, target_var)
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * 0.0001
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        lr = 0.001
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(
                loss, params, learning_rate=sh_lr, momentum=0.9)
        
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = dice_coef_loss(target_var, prediction)
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
    #                  dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_prediction])

    if model is None:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch, i in izip(iterate_minibatches(X_train, Y_train, 32, shuffle=True), tqdm(range(X_train.shape[0]/32))):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            import pdb; pdb.set_trace()
            for batch in iterate_minibatches(X_test, Y_test, 32, shuffle=False):
                inputs, targets = batch
                err, prediction = val_fn(inputs, targets)
                val_err += err
                val_batches += 1

            import pdb; pdb.set_trace()
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            # adjust learning rate as in paper
            # 32k and 48k iterations should be roughly equivalent to 41 and 61 epochs
            if (epoch+1) == 41 or (epoch+1) == 61:
                new_lr = sh_lr.get_value() * 0.1
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

        # dump the network weights to a file :
        np.savez('deep_residual_model.npz', *lasagne.layers.get_all_param_values(network))
    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, prediction = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))



    '''
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std
    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    '''

if __name__ == '__main__':
    train_and_predict()
