from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
from custom_inception_model import get_model
from data import load_train_data, load_test_data, random_crops
from learning_rate_reducer import SimpleLrReducer

img_rows = 128
img_cols = 160

smooth = 1.
import sys
sys.setrecursionlimit(10000)

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

    imgs_train = imgs_train[:30]
    imgs_mask_train = imgs_mask_train[:30]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_model(img_rows, img_cols)
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #lrreducer = SimpleLrReducer(5, .94)
    model.fit(imgs_train, imgs_mask_train, batch_size=5, nb_epoch=250, verbose=1, shuffle=True,
              callbacks=[model_checkpoint], validation_split=0.15)

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
