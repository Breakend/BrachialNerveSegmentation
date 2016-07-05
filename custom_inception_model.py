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
USE_BN = False  # whether to use batch normalization
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
    # total_branch = Lambda(lambda y: y * .1)(total_branch)
    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 
    return x

def inception_resnet_b(x):
    # mixed4: 17 x 17 x 768
    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch7x7 = conv2D_bn(x, 128, 1, 1)
    branch7x7 = conv2D_bn(branch7x7, 192, 7, 7)

    total_branch = merge([branch1x1, branch7x7], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, x._keras_shape[1], 1, 1, activation='linear')
    #    total_branch = Lambda(lambda y: y * .1)(total_branch)

    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 

    return x

def inception_resnet_c(x):
    branch1x1 = conv2D_bn(x, 192, 1, 1)

    branch3x3 = conv2D_bn(x, 192, 1, 1)
    branch3x3 = conv2D_bn(branch3x3, 256, 3, 3)

    total_branch = merge([branch1x1, branch3x3], mode='concat', concat_axis=CONCAT_AXIS)
    total_branch = conv2D_bn(total_branch, x._keras_shape[1], 1, 1, activation='linear')
    #    total_branch = Lambda(lambda y: y * .1)(total_branch)

    x = merge([total_branch, x], mode='sum')
    x = Activation('relu')(x) 

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

    x = modded_stem(img_input)
    
    for i in range(0,2):
        x = inception_resnet_a(x)

    x = MaxPooling2D((2,2))(x)

    for i in range(0,5):
        x = inception_resnet_b(x)

    x = MaxPooling2D((2,2))(x)

    #import pdb; pdb.set_trace()
    for i in range(0,2):
        x = inception_resnet_c(x)

    x = conv2D_bn(x, 1024, 1, 1, activation='relu', border_mode='same')
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = Dense(img_rows/4 * img_cols/4)(x)
    x = Reshape((1, img_rows/4, img_cols/4))(x)
    x = UpSampling2D((4,4))(x) 

    x = conv2D_bn(x, 1, 1, 1, activation='sigmoid', border_mode='same')

    # Define model

    model = Model(input=img_input, output=x)

    model.compile(optimizer=Nadam(), loss=dice_coef_loss, metrics=[dice_coef])
    #model.compile(optimizer=RMSprop(lr=.045, rho=0.9, epsilon=1.), loss=dice_coef_loss, metrics=[dice_coef])

    return model
