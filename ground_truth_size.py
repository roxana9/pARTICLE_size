# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:02:21 2021

@author: Roxana
"""
from matplotlib import pyplot as plt
import scale_detection as s 
import cv2
import glob
import imutils
from keras.models import load_model
smooth=1
import numpy as np
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import PIL
import keras
import pickle
#replace with your data_ground_truth path
path='TEST_FINAL/16/ground_truth/*.png'
files = glob.glob(path)
size_of_image=[]
Scales=[]
A=[]
for f in files:
    scale = cv2.imread(f, cv2.IMREAD_COLOR)
    img_scale=s.scale_line(scale).scalebar()
    Scales.append(img_scale)
#Scales=[4.545454545454546e-09, 3.508771929824562e-09, 7.352941176470588e-09, 1.785714285714286e-09, 4.651162790697674e-07, 2.272727272727273e-09, 8.928571428571428e-09, 1.1111111111111113e-09, 7.194244604316547e-10, 3.508771929824562e-09, 3.508771929824562e-09, 3.508771929824562e-09, 6.896551724137932e-09]
i=0
for f in files:
    img= cv2.imread(f, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    U=np.array(gray)
    thresh,binary_image = cv2.threshold(U,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    count_white = np.sum(binary_image > 0)
    count_black = np.sum(binary_image == 0)
    if count_white > count_black :
        binary_image = 255 - binary_image
    cnts = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,
     	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    contour_area=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if area==0:
            area=len(c)
        area=area*Scales[i]
     	# compute the center of the contour
     	# draw the contour and center of the shape on the image
        cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
        contour_area.append(area)
    tup=tuple(contour_area)
    size_of_image.append(tup)
    i=i+1
with open("test.txt", "wb") as fp:   #Pickling
    pickle.dump(size_of_image, fp)