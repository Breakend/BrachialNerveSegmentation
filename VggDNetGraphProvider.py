from keras.models import *
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.regularizers import l2


# TODO: this is no longer VGG
class VggDNetGraphProvider(object):
    def get_vgg_partial_graph(self, img_rows, img_cols):
        model = Sequential()
        #model.add_input(name='input', input_shape=(1, img_rows, img_cols))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001), input_shape=(1,img_rows, img_cols)))
        model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(Convolution2D(512, 3, 3, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        #TODO: try: http://cs231n.stanford.edu/reports2016/318_Report.pdf
        model.add(Convolution2D(1024, 1, 1, activation='relu', border_mode='same', W_regularizer=l2(0.00001)))
        model.add(UpSampling2D((32,32)))
        model.add(Convolution2D(1, 1, 1, activation='sigmoid'))
        #model.add(Flatten())
        #model.add(Dense(img_rows/2*img_cols/2))
        #model.add(Dropout(0.5))
        #model.add(Dense(img_rows/2*img_cols/2))
        #model.add(Dropout(0.5))
        #model.add(Dense(img_rows/2*img_cols/2))
        #model.add(Reshape((1, img_rows/2, img_cols/2)))
        #model.add(UpSampling2D((2,2)))

        return model
