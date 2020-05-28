import os
import glob
import cv2
from mtcnn import MTCNN
import argparse
# Select GPU before generation
os.environ["CUDA_DEVICE_ORDER"] = "0"


def data_source(name):
    if name == 'youtube':
        source_name = 'youtube'
        source_dir = os.path.join(root_dir, 'videos/original_sequences/youtube/raw/videos')
    elif name == 'Deepfakes':
        source_name = 'Deepfakes'
        source_dir = os.path.join(root_dir, 'videos/manipulated_sequences/Deepfakes/raw/videos')
    elif name == 'Face2Face':
        source_name = 'Face2Face'
        source_dir = os.path.join(root_dir, 'videos/manipulated_sequences/Face2Face/raw/videos')

    return source_name, source_dir


def video2image(name, video_dir):
    save_dir = os.path.join(root_dir, 'images', name)

    if not(os.path.isdir(save_dir)):
        os.makedirs(save_dir)

    os.chdir(video_dir)

    video_list = glob.glob("*")

    for video in video_list:
        os.chdir(video_dir)

        cap = cv2.VideoCapture(video)

        os.chdir(save_dir)

        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break

            cv2.imwrite(video + str(i) + '.jpg', frame)
            i += 1

        cap.release()
        cv2.destroyAllWindows()

    os.chdir(root_dir)
    return save_dir


def crop_face(img_dir):
    detector = MTCNN()

    os.chdir(img_dir)

    image_list = glob.glob("*")

    for i in image_list:
        image = cv2.imread(i)
        image2 = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(image)

        try:
            bounding_box = result[0]['box']
        except IndexError:
            os.remove(i)
            continue

        keypoints = result[0]['keypoints']

        nose = keypoints['nose']

        box_length_1 = bounding_box[2]
        box_length_2 = bounding_box[3]

        if box_length_1 >= box_length_2:
            box_length = int(box_length_1 / 2)
        else:
            box_length = int(box_length_2 / 2)

        cropped_image = image2[(nose[1] - box_length):(nose[1] + box_length),
                                (nose[0] - box_length):(nose[0] + box_length)]

        if cropped_image.size == 0:
            continue
        else:
            cv2.imwrite(i, cropped_image)

    os.chdir(root_dir)


# root directory of the dataset folder
root_dir = os.getcwd()

parser = argparse.ArgumentParser(description='Preprocess video to image')
parser.add_argument('-d', required=True, type=str, help='Choose the video dataset youtube, Deepfakes, Face2Face')

args = parser.parse_args()

# obtain source data name, directory from input
data_name, data_dir = data_source(args.d)

# cut the video by frame
image_dir = video2image(data_name, data_dir)
print("Cutting out frames from videos is complete.")

# crop face part from image
crop_face(image_dir)
print("Cropping out faces from images is complete.")
