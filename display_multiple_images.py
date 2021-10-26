import cv2
import numpy as np
from matplotlib import pyplot as plt 
import glob #import glob library 

#use glob to load image files from path
img = [cv2.imread(file) for file in glob.glob('/home/darnel/catkin_ws/src/VINS-Mono/rosbag_save/320x240_25Hz/*.jpg')] 
#print the matrix of the images
print("image matrix", img) 
#assign the images to a "list" and each frame of image, cv2.imshow the image

for i in img:
    cv2.imshow('image',i)
   
cv2.waitKey(0)
   
