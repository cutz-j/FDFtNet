import os
import argparse
import numpy as np
from tqdm import trange
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# argparse
parser = argparse.ArgumentParser(description='Model evaluation')

parser.add_argument('-fdft_model', required=True, type=str, help='select the pre-trained network')
parser.add_argument('-network', required=True, type=str, help='select the backbone network')
parser.add_argument('-test_dir', required=True, type=str, help='train image directory')
parser.add_argument('-img_height', type=str, default=64, help='image height')
parser.add_argument('-img_width', type=int, default=64, help='image width')
parser.add_argument('-batch_size', type=int, default=128, help='batch_size')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(args.test_dir,
                                                  target_size=(args.img_height, args.img_width),
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

model_fdft = load_model(args.ftft_model)

output_score = []
output_class = []
answer_class = []
answer_class_1 = []

for i in trange(len(test_generator)):
    output = model_fdft.predict_on_batch(test_generator[i][0])
    output_score.append(output)
    answer_class.append(test_generator[i][1])

output_score = np.concatenate(output_score)
answer_class = np.concatenate(answer_class)

output_class = np.argmax(output_score, axis=1)
answer_class_1 = np.argmax(answer_class, axis=1)

print(output_class)
print(answer_class_1)

cm = confusion_matrix(answer_class_1, output_class)
report = classification_report(answer_class_1, output_class)

recall = cm[0][0] / (cm[0][0] + cm[0][1])
fallout = cm[1][0] / (cm[1][0] + cm[1][1])

fpr, tpr, thresholds = roc_curve(answer_class_1, output_score[:, 1], pos_label=1.)
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thresh = interp1d(fpr, thresholds)(eer)

print(report)
print(cm)
print("AUROC: %f" %(roc_auc_score(answer_class_1, output_score[:, 1])))
print(thresh)
print('test_acc: ', len(output_class[np.equal(output_class, answer_class_1)]) / len(output_class))