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
from keras.layers.core import Activation
from keras.layers import Input, merge, ZeroPadding2D, Lambda
from keras.models import Model
from keras import regularizers
from Nadam import Nadam
from keras.optimizers import Adam, RMSprop
from keras import backend as K


smooth = 1.
# global constants
NB_CLASS = 1000  # number of classes
DIM_ORDERING = 'th'  # 'th' (channels, width, height) or 'tf' (width, height, channels)
WEIGHT_DECAY = None #0.000001  # L2 regularization factor
USE_BN = True  # whether to use batch normalization
if DIM_ORDERING == 'th':
    CONCAT_AXIS = 1
elif DIM_ORDERING == 'tf':
    CONCAT_AXIS = 3
 
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

def reduction_a(x, k, l, m, n):
    branch3x3 = conv2D_bn(x, n, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2D_bn(x, k, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, l, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, m, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)
    return x

# Inception-A blocks, 35x35
def inception_resnet_a(x):
    branch1x1 = conv2D_bn(x, 32, 1, 1)

    branch3x3 = conv2D_bn(x, 32, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 32, 3, 3)

    branch3x3dbl = conv2D_bn(x, 32, 1, 1)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 48, 3, 3)
    branch3x3dbl = conv2D_bn(branch3x3dbl, 64, 3, 3)

    total_branch = merge([branch1x1, branch3x3, branch3x3dbl], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, x._keras_shape[1], 1, 1, activation='linear')
    total_branch = Lambda(lambda y: y * .1)(total_branch)
    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 
    return x

def inception_resnet_b(x):
    # mixed4: 17 x 17 x 768
    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 128, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 160, 1, 7)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 1)

    total_branch = merge([branch1x1, branch7x7], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, x._keras_shape[1], 1, 1, activation='linear')
    total_branch = Lambda(lambda y: y * .1)(total_branch)

    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 

    return x

def reduction_b(x):
    branch3x3 = conv2D_bn(x, 256, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3_2 = conv2D_bn(x, 256, 1, 1)
    branch3x3_2 = conv2D_bn(branch3x3_2, 288, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3_3 = conv2D_bn(x, 256, 1, 1)
    branch3x3_3 = conv2D_bn(branch3x3_3, 288, 3, 3)
    branch3x3_3 = conv2D_bn(branch3x3_3, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering=DIM_ORDERING)(x)
    x = merge([branch3x3, branch3x3_2, branch3x3_3, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)
    return x

def inception_resnet_c(x):
    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch3x3 = conv2D_bn(x, 192, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 224, 1, 3)
    branch3x3 = conv2D_bn(branch3x3, 256, 3, 1)

    total_branch = merge([branch1x1, branch3x3], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, x._keras_shape[1], 1, 1, activation='linear')
    total_branch = Lambda(lambda y: y * .1)(total_branch)

    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 

    return x

def inception_stem(img_input):
    x = conv2D_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2D_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2D_bn(x, 64, 3, 3, border_mode='same')
    split1 = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING, border_mode='valid')(x)
    split2 = conv2D_bn(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')
    x = merge([split1, split2], mode='concat', concat_axis=CONCAT_AXIS)

    # left branch
    split1 = conv2D_bn(x, 64, 1, 1, border_mode='same')
    split1 = conv2D_bn(split1, 96, 3, 3, border_mode='valid')

    #right branch
    split2 = conv2D_bn(x, 64, 1, 1, border_mode='same')
    split2 = conv2D_bn(split2, 64, 7, 1, border_mode='same')
    split2 = conv2D_bn(split2, 64, 1, 7, border_mode='same')
    split2 = conv2D_bn(split2, 96, 3, 3, border_mode='valid')
    x = merge([split1, split2], mode='concat', concat_axis=CONCAT_AXIS)

    split1 = conv2D_bn(x, 192, 3, 3, subsample=(2,2), border_mode='valid')
    split2 = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=DIM_ORDERING, border_mode='valid')(x)
    x = merge([split1, split2], mode='concat', concat_axis=CONCAT_AXIS)

    return x

def modded_stem(img_input):
    x = conv2D_bn(img_input, 64, 3, 3)
    x = conv2D_bn(x, 64, 3, 3)
    x = MaxPooling2D((2, 2))(x)
    x = conv2D_bn(x, 128, 3, 3)
    x = conv2D_bn(x, 128, 3, 3)
    x = MaxPooling2D((2, 2))(x)
    x = conv2D_bn(x, 256, 3, 3)
    x = conv2D_bn(x, 256, 3, 3)
    return x

def get_model(img_rows, img_cols, channels=1):
    # Define image input layer

    if DIM_ORDERING == 'th':
        img_input = Input(shape=(channels, img_rows, img_cols))
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        img_input = Input(shape=(img_rows, img_cols, channels))
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Stem
    # Reduction-A

    x = inception_stem(img_input)
    
    for i in range(0,5):
        x = inception_resnet_a(x)

    x = reduction_a(x, k=256, l=256, m=384, n=384)

    for i in range(0,10):
        x = inception_resnet_b(x)

    x = reduction_b(x)

    #import pdb; pdb.set_trace()
    for i in range(0,5):
        x = inception_resnet_c(x)
    
    x = AveragePooling2D((2, 3), strides=(1, 1), dim_ordering=DIM_ORDERING)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(img_rows/2 * img_cols/2)(x)
    x = Reshape((1, img_rows/2, img_cols/2))(x)
    x = UpSampling2D((2,2))(x) 

    x = conv2D_bn(x, 1, 1, 1, activation='sigmoid', border_mode='same')

    # Define model

    model = Model(input=img_input, output=x)

    model.compile(optimizer=Adam(1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=RMSprop(lr=.045, rho=0.9, epsilon=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model
