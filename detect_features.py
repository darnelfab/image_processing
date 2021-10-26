import cv2
import numpy as np
from matplotlib import pyplot as plt
#read image from folder 
img = cv2.imread('/home/darnel/catkin_ws/src/VINS-Mono/rosbag_save/320x240_25Hz/frame0615.jpg')
#convert image to grayscale
gray_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#convert uint8 image to float32 normalised image
gray_img = np.float32(gray_img1)
#Harris corner detector
dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
#dst: output array, #gray_img: input single-channel image, blockSize: neighbourhood size, 
# ksize: aperture parameter for the Sobel operator, k: Harris detector free parameter
# dilate to make the corners more visible in the foreground
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 255, 0]
cv2.imshow('haris_corner', img)
#perform laplacian filter using laplacian kernel to detect blur
laplacian_var = cv2.Laplacian(gray_img1, cv2.CV_64F).var()
print(laplacian_var)
#obtain the histogram of the image
hist = cv2.calcHist([gray_img1],[0],None,[256],[0,256])
plt.hist(gray_img1.ravel(),256,[0,256])
plt.show()

#Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray_img1, maxCorners=50,      qualityLevel=0.02, minDistance=20)
corners = np.float32(corners)

for item in corners:
    x, y = item[0]
    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
cv2.imshow('Shi-Tomasi corner detection', img)

"""
AttributeError: module 'cv2' has no attribute 'xfeatures2d'
#SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray_img, None)
kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT', kp_img)
"""

"""
AttributeError: module 'cv2' has no attribute 'xfeatures2d'
#SURF
surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SURF', kp_img)
"""

"""
#FAST
fast = cv2.FastFeatureDetector_create()
fast.setNonmaxSuppression(False)
kp = fast.detect(gray_img, None)
kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
cv2.imshow('FAST', kp_img)
"""
#ORB
img = cv2.imread('images/scene.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)
kp, des = orb.detectAndCompute(gray_img, None)

kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)

cv2.imshow('ORB', kp_img)

cv2.waitKey()

