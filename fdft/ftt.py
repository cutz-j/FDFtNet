from utils import *
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.models import Model, load_model, model_from_json
from keras.regularizers import l2


def att(M=3, shape=(64, 64, 3), ld=1e-5):
######### Image Attention Model #########
### Block 1 ###
    x = Input(shape)  # (12, 12, 32)
    x3 = x
    for m in range(1, M+1):
        x3 = SeparableConv2D(32*m, kernel_size=(3, 3), strides=(2, 2), padding='same', depthwise_regularizer=l2(ld), pointwise_regularizer=l2(ld), use_bias=False)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Attention(32*m)(x3)


    ### final stage ###
    x6 = Conv2D(576, kernel_size=1, strides=1, padding='valid', kernel_regularizer=l2(ld), use_bias=False)(x3)
    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = GlobalAveragePooling2D()(x6)

    model = Model(inputs=x, outputs=x6)
    return model
