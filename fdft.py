from utils import *
from fdft import *
from network import *
import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Input
from keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Dropout
from keras.layers import SeparableConv2D, Add, Convolution2D, concatenate, Layer, ReLU, DepthwiseConv2D, Reshape, Multiply, InputSpec
from keras.regularizers import l2
from keras.models import Model


"""
ft_dir
shear_range
zoom_range
rotation_range
width_shift_range
height_shift_range
horizontal_flip
zca_whitening
cutout_use (boolean)
if cutout_use:
    co = cutout
else:
    co = None

img_height
img_width
batch_size
validation_dir
ld
"""
img_height = 64
img_width = 64

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
### 나중에 지우기



##

ft_dir = '/mnt/a/fakedata/deepfake/finetune'
train_gen_aug = ImageDataGenerator(shear_range=0,
                               zoom_range=0,
                               rotation_range=0.2,
                               width_shift_range=2.,
                               height_shift_range=2.,
                               horizontal_flip=True,
                               zca_whitening=False,
                               fill_mode='nearest',
                               preprocessing_function=cutout) # chagne this to co

test_datagen = ImageDataGenerator(rescale=1./255)

ft_gen = train_gen_aug.flow_from_directory(ft_dir,
                                              target_size=(img_height, img_width),
                                              batch_size=batch_size,
                                              shuffle=True,
                                              class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')


model_ft = load_model('/home/www/fake_detection/model/deepfake_xception.h5') # change this to weight
for i in range(2):
    model_ft.layers.pop()
im_in = Input(shape=(img_width, img_height, 3))

if --#xception, resnet,squeezenet
    base_model = xception(include_top=False)
    base_model.set_weights(model_ft.get_weights())
# for i in range(len(base_model.layers) - 0):
#     base_model.layers[i].trainable = False

pt_output = base_model(im_in)

mb = block(shape=tf.Tensor.get_shape(pt_output)[1:])
ftt = att(shape=(img_width, img_height, 3))

mb_output = mb(pt_output)
ftt_output = ftt(im_in)
######## final addition #########

x2 = Add()([mb_output, ftt_output])
x2 = Dense(2, kernel_regularizer=l2(1e-5))(x2)
x2 = Activation('softmax')(x2)

model_top = Model(inputs=im_in, outputs=x2)
model_top.summary()

# optimizer = SGD(lr=1e-3, momentum=0.9, nesterov=True)
optimizer = Adam()
model_top.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
callback_list = [EarlyStopping(monitor='val_acc', patience=es_patieance),
                 ReduceLROnPlateau(monitor='loss', factor=reduce_factor, cooldown=0, patience=5, min_lr=0.5e-5)]
output = model_top.fit_generator(ft_gen, steps_per_epoch=200, epochs=300,
                                  validation_data=validation_generator, validation_steps=len(validation_generator), callbacks=callback_list)
