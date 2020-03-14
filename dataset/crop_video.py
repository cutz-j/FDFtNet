import os
import video_frame, crop_image

main_dir = os.getcwd()

print("Starting video to frame images")

video_frame()

print("Video to frame images complete")

print("Starting cropping face from images")

os.chdir(main_dir)

crop_image()

print("Cropping face from images complete")
