from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam
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

def get_unet(dropout=.5):
    # TODO: try adding batch_norm to these between the activation and the conv2d layers
    # TODO: add dropout layers?
    inputs = Input((1, img_rows, img_cols))

    bigger_conv1 = Convolution2D(32, 11, 11, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(inputs)
    bigger_conv1 = Convolution2D(32, 11, 11, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_conv1)
    bigger_pool1 = MaxPooling2D(pool_size=(2, 2))(bigger_conv1)

    bigger_conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_pool1)
    bigger_conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_conv2)
    bigger_pool2 = MaxPooling2D(pool_size=(2, 2))(bigger_conv2)

    bigger_conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_pool2)
    bigger_conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_conv3)
    bigger_pool3 = MaxPooling2D(pool_size=(2, 2))(bigger_conv3)

    bigger_conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_pool3)
    bigger_conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(bigger_conv4)
    bigger_pool4 = MaxPooling2D(pool_size=(2, 2))(bigger_conv4)

    med_conv1 = Convolution2D(32, 7, 7, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(inputs)
    med_conv1 = Convolution2D(32, 7, 7, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_conv1)
    med_pool1 = MaxPooling2D(pool_size=(2, 2))(med_conv1)

    med_conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_pool1)
    med_conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_conv2)
    med_pool2 = MaxPooling2D(pool_size=(2, 2))(med_conv2)

    med_conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_pool2)
    med_conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_conv3)
    med_pool3 = MaxPooling2D(pool_size=(2, 2))(med_conv3)

    med_conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_pool3)
    med_conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(med_conv4)
    med_pool4 = MaxPooling2D(pool_size=(2, 2))(med_conv4)

    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    pool4 = merge([bigger_pool4, pool4, med_pool4], mode='concat', concat_axis=1)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv5)

    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(pool5)
    conv6 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv5], mode='concat', concat_axis=1)
    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up7)
    conv7 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv7)

    larger_conv7 = Convolution2D(512, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up7)
    larger_conv7 = Convolution2D(512, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(larger_conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), UpSampling2D(size=(2,2))(larger_conv7), conv4], mode='concat', concat_axis=1)
    conv8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up8)
    conv8 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv8)

    larger_conv8 = Convolution2D(256, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up8)
    larger_conv8 = Convolution2D(256, 5, 5, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(larger_conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), UpSampling2D(size=(2, 2))(larger_conv8), conv3], mode='concat', concat_axis=1)
    conv9 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up9)
    conv9 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv9)

    up10 = merge([UpSampling2D(size=(2, 2))(conv9), conv2], mode='concat', concat_axis=1)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up10)
    conv10 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv10)

    up11 = merge([UpSampling2D(size=(2, 2))(conv10), conv1], mode='concat', concat_axis=1)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(up11)
    conv11 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00013))(conv11)

    conv12 = Convolution2D(1, 1, 1, activation='sigmoid')(conv11)

    model = Model(input=inputs, output=conv12)

    # I might've done it wrong, but l2 made it perform worse. I think the l2 weights were overpowering the loss function though so maybe next time i can retry with < .001
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


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
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=3)
    model.fit(imgs_train, imgs_mask_train, batch_size=5, nb_epoch=50, verbose=1, shuffle=True,
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
