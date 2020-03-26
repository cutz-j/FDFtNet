import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.applications import ResNet152


def resNetV2(img_width=64, img_height=64):
    model = ResNet152(include_top=True, weights=None, input_shape=(img_width, img_height, 3), classes=2)
    return model
