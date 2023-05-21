# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 15:28:23 2021

@author: Roxana
"""
import cv2
import numpy as np
import easyocr
class scale_line():
    """return scalebar in suitable unit
    scale=unit*pixel /length
    """
    def __init__ (self,img):
        self.image=img
    def Line_detection(self):
        scale_gray = cv2.cvtColor(self.scale, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(scale_gray, 210, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filter noisy detection
        contours = [c for c in contours if cv2.contourArea(c) > 4500]
        # sort from by (y, x)
        contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        # work on the segment
        cv2.rectangle(self.scale, cv2.boundingRect(contours[-1]), (0,255,0), 1)
        x,y,w,h = cv2.boundingRect(contours[-1])
        # b=np.round(w/2);b=int(b)
        x=x+5;y=y+3;h=h-5;
        img=self.scale[y:y+h, x:x+260]
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        #####line calculation in pixel
        # img =  img[600:768, 0:130]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        base = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=2, maxLineGap=2)
        pixel_array = []
        pixel_length = 0
        if base is not None:
            for line in base:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                pixel_length = np.abs(x2 - x1)
                pixel_array.append(pixel_length)
                print("Pixel Length: {}".format(pixel_length))
        print("Average Pixel Length: {:.0f} pixel".format(np.max(pixel_array)))
        
        return (max(pixel_array),gray)
        
    def scalebar(self):
        self.scale=self.image
        pixel_l,gray=self.Line_detection()
        #text_and_digit_detection
        print('loading text detection algorithm')
        _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            
            # invert the image if the text is white and background is black
        count_white = np.sum(binary_image > 0)
        count_black = np.sum(binary_image == 0)
        if count_black > count_white:
           binary_image = 255 - binary_image
        reader=easyocr.Reader(['en'])
        bound=reader.readtext(binary_image)
        print(bound[0][1])
        ###################################S is the conveted scale to nm or um
        if len(bound[0][1])>2:
            temp=bound[0][1].split(' ')
            if temp[1]=='nm':
                S=(1e-9)*int(temp[0])
            else:
                S=(1e-6)*int(temp[0])
        elif bound[0][1]== 'nm':
            S=(1e-9)
        else:
            S=(1e-6)
        scale_f=S/pixel_l
        return scale_f      
        
       
       
       