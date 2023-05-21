#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 22:12:01 2023

@author: roxana
"""

import os
import PIL
from PIL import ImageOps
import numpy as np
from keras.layers import Input, Conv2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
import cv2
import keras
import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
print("loading dataset")
#replace with your data path
file_path='/kaggle/input/particle-array-2/dataset_array_2'
X_train=np.load('file_path/x_train_gray')
mask_train =np.load('file_path/mask_train')
X_val =np.load('file_path/x_val_gray')
mask_val=np.load('file_path/mask_val')
X_test=np.load('file_path/x_test_gray')
mask_test=np.load('file_path/mask_test')
#preprocessing_step
X_train=np.reshape(X_train,(len(X_train),256,256,1))
mask_train=np.reshape(mask_train,(len(mask_train),256,256,1))
X_val=np.reshape(X_val,(len(X_val),256,256,1))
mask_val=np.reshape(mask_val,(len(mask_val),256,256,1))
X_test=np.reshape(X_test,(len(X_test),256,256,1))
mask_test=np.reshape(mask_test,(len(mask_test),256,256,1))
img_size = (256,256)
num_classes = 2
batch_size = 16
print("training")
smooth=1
#you can use other backbones
BACKBONE = 'vgg16'
preprocess_input = sm.get_preprocessing(BACKBONE)

base_model = sm.Linknet(backbone_name='vgg16', encoder_weights=None)
#uncomment the following line when you intend to use pretrained weights
# base_model = sm.Linknet(BACKBONE, encoder_weights='imagenet',input_shape=(None, None, 3))
# for layer in base_model.layers:
#   if layer.name == 'sigmoid':
#     break
#   layer.trainable = False
 
inp = Input(shape=(None, None, 1))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)
model.summary()

from tensorflow.keras.optimizers import Adam, RMSprop,SGD
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score,'accuracy'])
# fit model
history=model.fit(
    x=X_train,
    y=mask_train,
    batch_size=16,
    epochs=500
)

score = model.evaluate(X_test,mask_test, verbose=1)
model_json = model.to_json()
with open("modelseg.json", "w") as json_file:
    json_file.write(model_json)

print("Saved model to disk")