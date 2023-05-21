# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:58:05 2021

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
loss=bce_jaccard_loss
dependencies = {
    'loss': loss,'iou_score':iou_score
}

#load the saved model
model=load_model('modelseg_new2_2.h5',compile=False)
#replace with your daata path
path='dataset/particles/test_frames_c/*.jpg'
files = glob.glob(path)
Scales=[]
A=[]; size_of_image=[]
for f in files:
    scale = cv2.imread(f, cv2.IMREAD_COLOR)
    img_scale=s.scale_line(scale).scalebar()
    Scales.append(img_scale)
    i=0
    image= cv2.imread(f, cv2.IMREAD_COLOR)
    resize=cv2.resize(image,(256,256))
    gray = cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
    gray=np.array(gray)
    gray=gray/255
    test_preds=gray.reshape(1,256,256,1)
    test_preds2 = model.predict(test_preds)
    p=test_preds2.reshape((256,256))
    mask = np.expand_dims(p, axis=-1)
    mask_array=keras.preprocessing.image.array_to_img(mask)
    img2 = PIL.ImageOps.autocontrast(mask_array)
    # # # display(img)
    img2.show()
    U=np.array(mask_array)
    thresh,binary_image = cv2.threshold(U,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    count_white = np.sum(binary_image > 0)
    count_black = np.sum(binary_image == 0)
    if count_white > count_black :
        binary_image = 255 - binary_image
    cnts = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL,
     	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    contour_area=[]
    c2=[]
    
    for c in cnts:
       # print(len(c))
        area=cv2.contourArea(c)
        if area==0:
            area=len(c)
        A.append(area)
        area=area*img_scale
        contour_area.append(area)
     	# compute the center of the contour
     	# draw the contour and center of the shape on the image
        cv2.drawContours(resize, [c], -1, (242,0,0), 1)
    cv2.imshow("img", resize)
    cv2.waitKey(0)   
    tup=tuple(contour_area)
    size_of_image.append(tup)
    i=i+1
    plt.imshow(resize)
    y_pos = np.arange(len(contour_area))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(y_pos,contour_area,color = 'b', width =0.5)
    plt.show()
# with open("test.txt", "wb") as fp:   #Pickling
#     pickle.dump(size_of_image, fp)
# with open("test.txt", "rb") as fp:   # Unpickling
# ...   b = pickle.load(fp)