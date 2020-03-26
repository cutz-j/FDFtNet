import numpy as np
from keras import backend as K

from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add
from keras.models import Model


def xception(img_height=64, img_width=64, dropout_rate=0.2):
    img_input = Input(shape=(img_height, img_width, 3))

    # layer 1 #
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='valid', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # layer 2 #
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # skip layer 1 #
    res = Conv2D(filters=128, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    # layer 3 #
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # layer 4 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = Add()([x, res])

    # skip layer 2 #
    res = Conv2D(filters=256, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    # layer 5 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # layer 6 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = Add()([x, res])

    # skip layer 3 #
    res = Conv2D(filters=728, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    # layer 7 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # layer 8 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = Add()([x, res])

    # ======== middle flow ========= #
    for i in range(8):
        # layer 9, 10, 11, 12, 13, 14, 15, 16, 17 #
        res = x

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = Add()([x, res])

        # ======== exit flow ========== #
    # skip layer 4 #
    res = Conv2D(filters=1024, kernel_size=(1, 1), strides=2, padding='same', use_bias=False)(x)
    res = BatchNormalization()(res)

    # layer 18 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=728, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # layer 19 #
    x = Activation('relu')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = Add()([x, res])

    # layer 20 #
    x = SeparableConv2D(filters=1536, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # layer 21 #
    x = SeparableConv2D(filters=2048, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2, activation='softmax')(x)
    output = Dropout(dropout_rate)(x)

    model = Model(img_input, output)
    return model
