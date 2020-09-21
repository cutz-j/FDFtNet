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
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# argparse
parser = argparse.ArgumentParser(description='Pretrain the models')

parser.add_argument('-pt_model', required=True, type=str, help='select the pre-trained network')
parser.add_argument('-network', required=True, type=str, help='select the backbone network')
parser.add_argument('-ft_dir', required=True, type=str, help='train image directory')
parser.add_argument('-val_dir', required=True, type=str, help='validation image directory')
parser.add_argument('-img_height', type=str, default=64, help='image height')
parser.add_argument('-img_width', type=int, default=64, help='image width')
parser.add_argument('-batch_size', type=int, default=128, help='batch_size')
parser.add_argument('-es_patience', type=int, default=20, help='early stopping patience')
parser.add_argument('-reduce_factor', type=int, default=0.1, help='reduce factor')
parser.add_argument('-reduce_patience', default=20, type=int, help='reduce patience')
parser.add_argument('-step', type=int, default=200, help='steps per epoch')
parser.add_argument('-epochs', type=int, default=300, help='epochs')
parser.add_argument('-dropout_rate', type=int, default=0.2, help='dropout rate')
parser.add_argument('-gpu_ids', type=str, default='0', help='select the GPU to use')

# augmentation
parser.add_argument('-shear_range', type=float, default=0, help='shear')
parser.add_argument('-zoom_range', type=float, default=0, help='zoom')
parser.add_argument('-rotation_range', type=float, default=0.2, help='rotation')
parser.add_argument('-width_shift_range', type=float, default=2.0, help='width shift')
parser.add_argument('-height_shift_range', type=float, default=2.0, help='height shift')
parser.add_argument('-horizontal_flip', type=bool, default=True, help='horizontal flip')
parser.add_argument('-zca_whitening', type=bool, default=False, help='zca_whitening')
parser.add_argument('-cutout_use', type=bool, default=True, help='zca_whitening')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# model selection
if args.network.lower() == 'xception':
    base_model = xception(args.img_height, args.img_weight, args.dropout_rate)
elif args.network.lower() == 'resnetv2':
    base_model = resNetV2(args.img_height, args.img_weight)
elif args.network.lower() == 'squeezenet':
    base_model = squeezeNet(args.img_height, args.img_weight, args.dropout_rate)


##
if args.cutout_use == False:
    cutout = None

ft_dir = args.ft_dir
train_gen_aug = ImageDataGenerator(shear_range=args.shear_range,
                               zoom_range=args.zoom_range,
                               rotation_range=args.rotation_range,
                               width_shift_range=args.width_shift_range,
                               height_shift_range=args.height_shift_range,
                               horizontal_flip=args.horizontal_flip,
                               zca_whitening=args.zca_whitening,
                               fill_mode='nearest',
                               preprocessing_function=cutout) # chagne this to co

test_datagen = ImageDataGenerator(rescale=1./255)

ft_gen = train_gen_aug.flow_from_directory(ft_dir,
                                              target_size=(args.img_height, args.img_width),
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              class_mode='categorical')


validation_generator = test_datagen.flow_from_directory(args.val_dir,
                                                        target_size=(args.img_height, args.img_width),
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        class_mode='categorical')


model_ft = load_model(args.pt_model) # change this to weight
for i in range(2):
    model_ft.layers.pop()
im_in = Input(shape=(args.img_width, args.img_height, 3))
base_model.set_weights(model_ft.get_weights())
# for i in range(len(base_model.layers) - 0):
#     base_model.layers[i].trainable = False

pt_output = base_model(im_in)

mb = block(shape=tf.Tensor.get_shape(pt_output)[1:])
ftt = att(shape=(args.img_width, args.img_height, 3))

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
callback_list = [EarlyStopping(monitor='val_accuracy', patience=args.es_patience),
                 ReduceLROnPlateau(monitor='val_loss', factor=args.reduce_factor, patience=args.reduce_patience)]
output = model_top.fit_generator(ft_gen, steps_per_epoch=args.step, epochs=args.epochs,
                                  validation_data=validation_generator, validation_steps=len(validation_generator), callbacks=callback_list)
