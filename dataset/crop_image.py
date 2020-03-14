import os
import glob
import cv2
from mtcnn import MTCNN

root_dir = os.getcwd()
image_dir = os.path.join(root_dir, 'images')
save_dir = os.path.join(root_dir, 'cropped_images')

detector = MTCNN()

os.chdir(image_dir)

image_list = glob.glob("*")

for i in image_list:
    os.chdir(image_dir)

    image = cv2.imread(i)
    image2 = image.copy()
    image_bound = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = detector.detect_faces(image)
    bounding_box = result[0]['box']

    cropped_image = image2[bounding_box[1]:(bounding_box[1] + bounding_box[3]), bounding_box[0]:(bounding_box[0] + bounding_box[2])]

    os.chdir(save_dir)

    cv2.imwrite(i + '_cropped.jpg', cropped_image)

os.chdir(root_dir)