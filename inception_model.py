'''This script demonstrates how to build the Inception v3 architecture
using the Keras functional API.
We are not actually training it here, for lack of appropriate data.

For more information about this architecture, see:

"Rethinking the Inception Architecture for Computer Vision"
Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
http://arxiv.org/abs/1512.00567
'''
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout, Reshape
from keras.layers import Input, merge, ZeroPadding2D
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras import backend as K
from Nadam import Nadam


# global constants
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = None #0.00001  # L2 regularization factor
USE_BN = True  # whether to use batch normalization


def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''Utility function to apply to a tensor a module conv + BN
    with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    return x


def get_model(img_rows, img_cols):
    # Define image input layer

    if DIM_ORDERING == 'th':
        img_input = Input(shape=(1, img_rows, img_cols))
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        img_input = Input(shape=(img_rows, img_cols, 1))
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Entry module

    layers = []

    x = conv2D_bn(img_input, 32, 3, 3, border_mode='same')
    x = conv2D_bn(x, 32, 3, 3, border_mode='same')
    x = conv2D_bn(x, 64, 3, 3, border_mode='same')
    x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = conv2D_bn(x, x._keras_shape[1], 2, 2, border_mode='valid')
    x = ZeroPadding2D((1,1), dim_ordering=DIM_ORDERING)(x)
 
    x = conv2D_bn(x, 80, 1, 1, border_mode='same')
    x = conv2D_bn(x, 192, 3, 3, border_mode='same')
    x = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = conv2D_bn(x, x._keras_shape[1], 2, 2, border_mode='valid')
    x = ZeroPadding2D((1,1), dim_ordering=DIM_ORDERING)(x)
 
    # mixed: 35 x 35 x 256

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed_1: 35 x 35 x 288

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed2: 35 x 35 x 288

    branch1x1 = conv2D_bn(x, 64, 1, 1)

    branch5x5 = conv2D_bn(x, 48, 1, 1)
    branch5x5 = conv2D_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)
    
    # mixed3: 17 x 17 x 768

    branch3x3 = conv2D_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2D_bn(x, 64, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    x = conv2D_bn(x, x._keras_shape[1], 2, 2, border_mode='valid')
    x = ZeroPadding2D((1,1), dim_ordering=DIM_ORDERING)(x)
 
    # mixed4: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 128, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 128, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed5: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed5: 17 x 17 x 768
    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed6: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 160, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 160, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed7: 17 x 17 x 768

    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 192, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2D_bn(x, 160, 1, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2D_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    branch3x3 = conv2D_bn(x, 192, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2D_bn(x, 192, 1, 1)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2D_bn(branch7x7x3, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch7x7x3, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)
    # mixed9: 8 x 8 x 2048

    x = conv2D_bn(x, x._keras_shape[1], 2, 2, border_mode='valid')
    x = ZeroPadding2D((1,1), dim_ordering=DIM_ORDERING)(x)

    branch1x1 = conv2D_bn(x, 320, 1, 1)

    branch3x3 = conv2D_bn(x, 384, 1, 1)
    branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2], mode='concat', concat_axis=CONCAT_AXIS)

    branch3x3dbl = conv2D_bn(x, 448, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2], mode='concat', concat_axis=CONCAT_AXIS)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    # mixed10: 8 x 8 x 2048
    branch1x1 = conv2D_bn(x, 320, 1, 1)

    branch3x3 = conv2D_bn(x, 384, 1, 1)
    branch3x3_1 = conv2D_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2D_bn(branch3x3, 384, 3, 1)
    branch3x3 = merge([branch3x3_1, branch3x3_2], mode='concat', concat_axis=CONCAT_AXIS)

    branch3x3dbl = conv2D_bn(x, 448, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2D_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2D_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2], mode='concat', concat_axis=CONCAT_AXIS)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORDERING)(x)
    branch_pool = conv2D_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)
    # Define model
    def dice_coef(y_true, y_pred):
        smooth = 1.0
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
        union = K.sum(y_true_f, axis=1, keepdims=True) + \
                K.sum(y_pred_f, axis=1, keepdims=True) + \
                smooth
        return K.mean(intersection / union)


    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    x = conv2D_bn(x, 512, 1, 1, activation='relu', border_mode='same')
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Dense(img_rows/2*img_cols/2)(x)
    x = Reshape((1, img_rows/2, img_cols/2))(x)

    x = UpSampling2D(size=(2, 2))(x)

    x = conv2D_bn(x, 1, 3, 3, activation='sigmoid', border_mode='same')

    model = Model(input=img_input, output=x)

    model.compile(optimizer=Nadam(), loss=dice_coef_loss, metrics=[dice_coef])

    return model
