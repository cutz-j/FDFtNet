import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.models import Model, load_model, model_from_json


def squeezeNet(img_height=64, img_width=64, dropout_rate=0.2, include_top=True):
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu = "relu_"

    # Modular function for Fire Node

    def fire_module(x, fire_id, squeeze=16, expand=64):
        s_id = 'fire' + str(fire_id) + '/'

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
        x = Activation('relu', name=s_id + relu + sq1x1)(x)

        left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
        left = Activation('relu', name=s_id + relu + exp1x1)(left)

        right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
        right = Activation('relu', name=s_id + relu + exp3x3)(right)

        x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
        return x

    # Original SqueezeNet from paper.

    img_input = Input(shape=(img_height, img_width, 3))

    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    x_dp = Dropout(dropout_rate)(x)
    x_conv = Convolution2D(2, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x_conv)

    if include_top == True:
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)

    model = Model(img_input, x, name='squeezenet')
    return model
