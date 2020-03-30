from utils import *
from keras.regularizers import l2
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.models import Model, load_model, model_from_json


def block(N=4, shape=(12, 12, 32), ld=1e-5):
    x1 = Input(shape) # (12, 12, 32)
    ########### Mobilenet block bottleneck 3x3 (32 --> 128) #################
    expand1 = Conv2D(576, kernel_size=1, strides=1, kernel_regularizer=l2(ld), use_bias=False)(x1)
    expand1 = BatchNormalization()(expand1)
    expand1 = HardSwish()(expand1)
    dw1 = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), padding='same', depthwise_regularizer=l2(ld), use_bias=False)(expand1)
    dw1 = BatchNormalization()(dw1)
    se_gap1 = GlobalAveragePooling2D()(dw1)
    se_gap1 = Reshape([1, 1, -1])(se_gap1)
    se1 = Conv2D(144, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(se_gap1)
    se1 = Activation('relu')(se1)
    se1 = Conv2D(576, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(se1)
    se1 = HardSigmoid()(se1)
    se1 = Multiply()([expand1, se1])
    project1 = HardSwish()(se1)
    project1 = Conv2D(128, kernel_size=(1, 1), padding='valid', kernel_regularizer=l2(ld), use_bias=False)(project1)
    project1 = BatchNormalization()(project1)

    for _ in range(N-1):
        ########### Mobilenet block bottleneck 5x5 (128 --> 128) #################
        expand2 = Conv2D(576, kernel_size=1, strides=1, kernel_regularizer=l2(ld), use_bias=False)(project1)
        expand2 = BatchNormalization()(expand2)
        expand2 = HardSwish()(expand2)
        dw2 = DepthwiseConv2D(kernel_size=(5,5), strides=(1,1), padding='same', depthwise_regularizer=l2(ld), use_bias=False)(expand2)
        dw2 = BatchNormalization()(dw2)
        se_gap2 = GlobalAveragePooling2D()(dw2)
        se_gap2 = Reshape([1, 1, -1])(se_gap2)
        se2 = Conv2D(144, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(se_gap2)
        se2 = Activation('relu')(se2)
        se2 = Conv2D(576, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(se2)
        se2 = HardSigmoid()(se2)
        se2 = Multiply()([expand2, se2])
        project2 = HardSwish()(se2)
        project2 = Conv2D(128, kernel_size=(1, 1), padding='valid', kernel_regularizer=l2(ld), use_bias=False)(project2)
        project2 = BatchNormalization()(project2)
        project2 = Add()([project1, project2])
        project1 = project2

    ########## Classification ##########
    x2 = Conv2D(576, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(project2)
    x2 = BatchNormalization()(x2)
    x2 = HardSwish()(x2)
    x2 = GlobalAveragePooling2D()(x2)

    model = Model(inputs=x1, outputs=x2)
    return model
