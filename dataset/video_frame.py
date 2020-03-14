import os
import glob
import cv2

root_dir = os.getcwd()
video_dir = os.path.join(root_dir, 'videos')
save_dir = os.path.join(root_dir, 'images')

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