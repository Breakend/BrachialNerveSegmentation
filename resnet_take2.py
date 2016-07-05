from __future__ import print_function

import cv2
import random
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, Activation, Flatten, Reshape, Dense, AveragePooling2D, ZeroPadding2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.regularizers import l2
import keras

import residual_blocks
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



def compute_padding_length(length_before, stride, length_conv):
    ''' Assumption: you want the subsampled result has a length of floor(original_length/stride).
    '''
    N = length_before
    F = length_conv
    S = stride
    if S == F:
        return 0
    if S == 1:
        return (F-1)/2
    for P in range(S):
        if (N-F+2*P)/S + 1 == N/S:
            return P
    return None

def design_for_residual_blocks(num_channel_input=1):
    ''''''
    model = Sequential() # it's a CONTAINER, not MODEL
    # set numbers
    num_big_blocks = 3
    image_patch_sizes = [[3,3]]*num_big_blocks
    pool_sizes = [(2,2)]*num_big_blocks
    n_features = [128, 256, 512, 512, 1024]
    n_features_next = [256, 512, 512, 512, 1024]
    height_input = img_rows
    width_input = img_cols
    for conv_idx in range(num_big_blocks):    
        n_feat_here = n_features[conv_idx]
        # residual block 0
        model.add(residual_blocks.building_residual_block(  (num_channel_input, height_input, width_input),
                                                            n_feat_here,
                                                            kernel_sizes=image_patch_sizes[conv_idx]
                                                            ))

        # residual block 1 (you can add it as you want (and your resources allow..))
        if False:
            model.add(residual_blocks.building_residual_block(  (n_feat_here, height_input, width_input),
                                                                n_feat_here,
                                                                kernel_sizes=image_patch_sizes[conv_idx]
                                                                ))
        
        # the last residual block N-1
        # the last one : pad zeros, subsamples, and increase #channels
        pad_height = compute_padding_length(height_input, pool_sizes[conv_idx][0], image_patch_sizes[conv_idx][0])
        pad_width = compute_padding_length(width_input, pool_sizes[conv_idx][1], image_patch_sizes[conv_idx][1])
        model.add(ZeroPadding2D(padding=(pad_height,pad_width))) 
        height_input += 2*pad_height
        width_input += 2*pad_width
        n_feat_next = n_features_next[conv_idx]
        model.add(residual_blocks.building_residual_block(  (n_feat_here, height_input, width_input),
                                                            n_feat_next,
                                                            kernel_sizes=image_patch_sizes[conv_idx],
                                                            is_subsample=True,
                                                            subsample=pool_sizes[conv_idx]
                                                            ))

        height_input, width_input = model.output_shape[2:]
        # width_input  = int(width_input/pool_sizes[conv_idx][1])
        num_channel_input = n_feat_next

    # Add average pooling at the end:
    #print('Average pooling, from (%d,%d) to (1,1)' % (height_input, width_input))
    #model.add(AveragePooling2D(pool_size=(height_input, width_input)))

    return model

def get_residual_model():
    model = keras.models.Sequential() # 
    model.add(ZeroPadding2D((2,2), input_shape=(1, img_rows, img_cols))) # resize (28,28)-->(32,32)
    # the first conv 
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    # [residual-based Conv layers]
    residual_blocks = design_for_residual_blocks(num_channel_input=128)
    model.add(residual_blocks)
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    print("Model shape after resnet")
    print(model.output_shape)

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense((img_rows/2) * (img_cols/2)))
    model.add(Reshape((1, img_rows/2, img_cols/2)))
    model.add(UpSampling2D((2,2)))
    model.add(Convolution2D(1, 1, 1, activation='sigmoid'))

    sgd = SGD(lr=0.1, decay=0.00005, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    return model



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
    model = get_residual_model()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(imgs_train, imgs_mask_train, batch_size=5, nb_epoch=1000, verbose=1, shuffle=True,
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
