from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
from keras.utils.visualize_util import plot

from data import load_train_data, load_test_data, random_crops

img_rows = 64
img_cols = 80

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet(dropout=.5):
    # TODO: try adding batch_norm to these between the activation and the conv2d layers
    # TODO: add dropout layers?
    inputs = Input((1, img_rows, img_cols))

    def __res_block(network, filter_size, kernel_size=3):
        _res_conv_1 = Convolution2D(filter_size, kernel_size, kernel_size, W_regularizer=l2(0.00005), border_mode='same')(network)
        _input_scaled_1 = Convolution2D(filter_size, 1, 1, border_mode='same', W_regularizer=l2(0.00005))(network)
        _res_batch_norm_1 = BatchNormalization(axis=1)(_res_conv_1) 
        _res_ac_1 = Activation('relu')(_res_batch_norm_1)
        _res_conv_2 = Convolution2D(filter_size, 3, 3, border_mode='same', W_regularizer=l2(0.00005))(_res_ac_1)
        _res_batch_norm_2 = BatchNormalization(axis=1)(_res_conv_2)
        _merged = merge([_res_batch_norm_2, _input_scaled_1], mode='concat', concat_axis=1)
        return Activation('relu')(_merged)

    res1 = __res_block(inputs, 64)
    med_res1 = __res_block(inputs, 64, 7)
    large_res1 = __res_block(inputs, 64, 11)

    layer1 = merge([res1, med_res1, large_res1], mode='concat', concat_axis=1)
    down1 = MaxPooling2D(pool_size=(2, 2))(layer1)    

    res2 = __res_block(down1, 128)
    med_res2 = __res_block(down1, 128, 7)
    large_res2 = __res_block(down1, 128, 11)

    layer2 = merge([res2, med_res2, large_res2], mode='concat', concat_axis=1)
    down2 = MaxPooling2D(pool_size=(2, 2))(layer2)    

    res3 = __res_block(down2, 256)
    down3 = MaxPooling2D(pool_size=(2,2))(res3)

    res4 = __res_block(down3, 512)
    down4 = MaxPooling2D(pool_size=(2,2))(res4)

    conv6 = Convolution2D(1024, 3, 3, activation='relu', W_regularizer=l2(0.00005), border_mode='same')(down4)
    conv6 = Convolution2D(1024, 3, 3, activation='relu', W_regularizer=l2(0.00005), border_mode='same')(conv6)

    up10 = merge([UpSampling2D(size=(8, 8))(conv6), layer2], mode='concat', concat_axis=1)
    res7 = __res_block(up10, 64)
    res7 = UpSampling2D((2,2))(res7)

    conv12 = Convolution2D(1, 1, 1, activation='sigmoid', W_regularizer=l2(0.00005))(res7)

    model = Model(input=inputs, output=conv12)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs, imgs_mask_train = None, number_augs_per_im = 5):
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
    plot(model, to_file='model.png')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
    model.fit(imgs_train, imgs_mask_train, batch_size=20, nb_epoch=30, verbose=1, shuffle=True,
              callbacks=[model_checkpoint, early_stopping], validation_split=0.10)

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
