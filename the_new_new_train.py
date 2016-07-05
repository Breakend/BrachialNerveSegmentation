from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, AveragePooling2D
from keras.layers import  Lambda, Flatten, Dense, RepeatVector, Reshape
from keras.optimizers import Adam, SGD
from keras.layers.core import Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
from Nadam import Nadam
from learning_rate_reducer import SimpleLrReducer

from preprocess_stuff import ImageDataGenerator

from data import load_train_data, load_test_data, random_crops

img_rows = 128
img_cols = 160

import sys
sys.setrecursionlimit(10000)

smooth = 1.
# global constants
NB_CLASS = 1000  # number of classes
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = None#0.00001  # L2 regularization factor
USE_BN = True  # whether to use batch normalization
if DIM_ORDERING == 'th':
    CONCAT_AXIS = 1
elif DIM_ORDERING == 'tf':
    CONCAT_AXIS = 3

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''Utility function to apply to a tensor a module conv + BN
    with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = l2(weight_decay)
        b_regularizer = l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='linear',
                      border_mode=border_mode,
                      init='he_normal',
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)

    if batch_norm:
        x = BatchNormalization(mode=0, axis=1)(x)

    if activation:
        x = Activation(activation)(x)
 
    return x


def down_block(x, samples, down=True):
    small_branch = conv2D_bn(x, samples, 1, 1)
    small_branch = conv2D_bn(small_branch, samples, 3, 3)
    small_branch = conv2D_bn(small_branch, samples, 3, 3, activation=None)

    med_branch = conv2D_bn(x, samples, 1, 1)
    med_branch = conv2D_bn(med_branch, samples, 5, 5)
    med_branch = conv2D_bn(med_branch, samples, 5, 5, activation=None)

    lrg_branch = conv2D_bn(x, samples, 1, 1)
    lrg_branch = conv2D_bn(lrg_branch, samples, 7, 7)
    lrg_branch = conv2D_bn(lrg_branch, samples, 7, 7, activation=None)

    total_branch = merge([small_branch, med_branch, lrg_branch], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, samples, 1, 1)
    total_branch = conv2D_bn(total_branch, samples, 1, 1, activation=None)
    #total_branch = conv2D_bn(total_branch, total_branch._keras_shape[1], 1, 1, activation='linear')
    #total_branch = Lambda(lambda y: y * .1)(total_branch)

    #total_branch = conv2D_bn(x, samples, 3, 3)
    #total_branch = conv2D_bn(total_branch, samples, 3, 3, activation=None)

    res_conn = conv2D_bn(x, total_branch._keras_shape[1], 1, 1)
    x = merge([total_branch, res_conn], mode='sum')

    if down:
        x = conv2D_bn(x, samples, 2, 2, subsample=(2,2), activation=None)

    return x

def upblock(x, samples, merge_block, up=True):
    total_branch = conv2D_bn(x, samples, 3, 3)
    total_branch = conv2D_bn(total_branch, samples, 3, 3, activation=None)

    res_conn = conv2D_bn(x, total_branch._keras_shape[1], 1, 1)
    x = merge([total_branch, res_conn], mode='sum')

    if merge_block: 
        merge_block = conv2D_bn(merge_block, samples, 1, 1)
        x = merge([UpSampling2D(size=(2, 2))(x), merge_block], mode='concat', concat_axis=1)
    elif up:
        x = UpSampling2D(size=(2, 2))(x)

    return x

def get_unet():
    inputs = Input((1, img_rows, img_cols))

    x = block1 = down_block(inputs, 64)
    x = block2 = down_block(x, 128)
    x = block3 = down_block(x, 256)
    x = block4 = down_block(x, 512)
    x = block5 = down_block(x, 1024)

    x = upblock(x, 1024, block4)
    x = upblock(x, 512, block3)
    x = upblock(x, 256, block2)
    x = upblock(x, 128, block1)
    x = upblock(x, 64, None)
    #x = upblock(x, 64, None, False)

    conv12 = Convolution2D(1, 1, 1, activation='sigmoid')(x)

    model = Model(input=inputs, output=conv12)

    model.compile(optimizer='rmsprop', loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    #imgs_train = imgs_train[:30]
    #imgs_mask_train = imgs_mask_train[:30]
    train_indices = np.arange(int(imgs_train.shape[0]*.85))
    np.random.shuffle(train_indices)
    X_train = imgs_train[train_indices]
    Y_train = imgs_mask_train[train_indices]
    Y_test = np.delete(imgs_mask_train, train_indices, 0)
    X_test = np.delete(imgs_train, train_indices, 0)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(imgs_train, imgs_mask_train)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    lrreducer = SimpleLrReducer(10, .5)
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
                        samples_per_epoch=len(imgs_train), nb_epoch=75, verbose=1,
                        callbacks=[model_checkpoint, lrreducer],
                        validation_data=(X_test, Y_test))

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
