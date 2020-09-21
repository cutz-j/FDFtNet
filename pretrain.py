from network import *
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# argparse
parser = argparse.ArgumentParser(description='Pretrain the models')

parser.add_argument('-network', required=True, type=str, help='select the backbone network')
parser.add_argument('-train_dir', required=True, type=str, help='train image directory')
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

args = parser.parse_args()

root_dir = os.getcwd()
weight_save_dir = os.path.join(root_dir, 'weights')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# model selection
if args.network.lower() == 'xception':
    model = xception(args.img_height, args.img_weight, args.dropout_rate)
elif args.network.lower() == 'resnetv2':
    model = resNetV2(args.img_height, args.img_weight)
elif args.network.lower() == 'squeezenet':
    model = squeezeNet(args.img_height, args.img_weight, args.dropout_rate)


model.summary()

# model compile
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(len(model.trainable_weights))

datagenerator = ImageDataGenerator(rotation_range=0.0,
                                   shear_range=0,
                                   zoom_range=0,
                                   width_shift_range=0,
                                   height_shift_range=0,
                                   horizontal_flip=False,
                                   rescale=1./255,)

train_generator = datagenerator.flow_from_directory(args.train_dir,
                                                    target_size=(args.img_height, args.img_width),
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    class_mode='categorical')

validation_generator = datagenerator.flow_from_directory(args.val_dir,
                                                         target_size=(args.img_height, args.img_width),
                                                         batch_size=args.batch_size,
                                                         shuffle=False,
                                                         class_mode='categorical')


callback_list = [EarlyStopping(monitor='val_accuracy', patience=args.es_patience),
                 ReduceLROnPlateau(monitor='val_loss', factor=args.reduce_factor, patience=args.reduce_patience)]

history = model.fit_generator(train_generator,
                              steps_per_epoch=args.step,
                              epochs=args.epochs,
                              validation_data=validation_generator,
                              validation_steps=len(validation_generator),
                              callbacks=callback_list)

# save the model weight
model.save(weight_save_dir)
