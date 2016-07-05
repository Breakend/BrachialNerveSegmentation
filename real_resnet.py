from __future__ import print_function

import cv2
import random
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation, Flatten, Reshape, Dense, AveragePooling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.regularizers import l2

from data import load_train_data, load_test_data, random_crops

img_rows = 128
img_cols = 160

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


def get_unet(dropout=.5):
    # TODO: try adding batch_norm to these between the activation and the conv2d layers
    # TODO: add dropout layers?
    inputs = Input((1, img_rows, img_cols))

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(inputs)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = _residual_block(block_fn, nb_filters=64, repetations=1, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=128, repetations=1)(block1)
    block3 = _residual_block(block_fn, nb_filters=256, repetations=1)(block2)
    block4 = _residual_block(block_fn, nb_filters=512, repetations=1)(block3)
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4) 
    conv2 = Convolution2D(1024, 3, 3, activation='relu', border_mode="same")(pool2)
    conv2 = Convolution2D(1024, 3, 3, activation='relu', border_mode="same")(conv2)
    conv2 = Convolution2D(512, 1, 1, activation='relu')(conv2)
    flat1 = Flatten()(conv2)
    dense1 = Dense((img_rows/2) * (img_cols/2))(flat1)
    dense1 = Dense((img_rows/2) * (img_cols/2))(dense1)
    dense1 = Reshape((1, img_rows/2, img_cols/2))(dense1)
    dense1 = UpSampling2D((2,2))(dense1)
    dense1 = Convolution2D(1, 1, 1, activation='sigmoid')(dense1)

    model = Model(input=inputs, output=dense1)

    sgd = SGD(lr=0.001, decay=0.00005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs, imgs_mask_train = None, number_augs_per_im = 0):
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
        percent_crop = random.uniform(.75, 1.0)
        au_img, au_msk = random_crops(imgs, imgs_mask_train, (int(imgs.shape[2] * percent_crop), int(imgs.shape[3]*percent_crop)))
        for i in range(imgs.shape[0]):
            imgs_p[i*(j+2), 0] = cv2.resize(au_img[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
            imgs_masks_p[i*(j+2), 0] = cv2.resize(au_msk[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)

    return imgs_p, imgs_masks_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train, imgs_mask_train = preprocess(imgs_train, imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(imgs_train, imgs_mask_train, batch_size=85, nb_epoch=1000, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, early_stopping], validation_split=0.15)

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


if __name__ == '__main__':
    train_and_predict()
