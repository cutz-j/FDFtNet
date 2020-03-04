import cv2
import os
import matplotlib.pyplot as plt

dataset_dir = "w:/home/kim1/IFIP/dataset/face2face/"
target_dir = "d:/dataset/df_rgb/"

t_list = os.listdir(dataset_dir)

for t in t_list:
    rf_list = os.listdir(dataset_dir+t)
    target_t = dataset_dir+t
    for rf in rf_list:
        target_rf = target_t + '/' + rf
        png_list = os.listdir(dataset_dir+t+'/'+rf)
        for png in png_list:
            png_dir = dataset_dir+t+'/'+rf+'/'+png
            target_png = target_rf+'/'+png
        
            bgr = cv2.imread(png_dir)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cv2.imwrite(target_png, rgb)
