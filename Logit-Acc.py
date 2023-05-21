# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 17:46:21 2021

@author: Roxana
"""
import pickle
import numpy as np
logit_acc=[];R=[];ground=[];proposed=[]
#the path of ground_truth_size
f_g2='TEST_FINAL/2/ground_truth/2_measure_g.txt'
#the path of predicted_size
f_p2='TEST_FINAL/2/2_measure.txt'
micro=[];nano=[]
with open(f_g2, "rb") as fp: 
    # Unpickling
    ground2= pickle.load(fp)

ground.extend(ground2)
with open(f_p2, "rb") as fp: 
    # Unpickling
    proposed2= pickle.load(fp)
proposed.extend(proposed2)

n=len(ground)
mean_p=[];mean_g=[];k=[];k2=[]
for i in range(n):
    temp_g=ground[i]
    temp_g=np.array(temp_g)
    temp_p=proposed[i]
    temp_p=np.array(temp_p)
    s_m=np.mean(temp_g)
    sigma_g=np.std(temp_g)
    mean_g.append(s_m)
    s_p=np.mean(temp_p)
    mean_p.append(s_p)
    sigma_p=np.std(temp_p)
    y1=sigma_p/s_p
    y2=sigma_g/s_m
    k.append(y1)
    k2.append(y2)
    if abs(s_m-s_p)<=sigma_p:
        logit_acc.append(1)
    else:
        logit_acc.append(0)
    R.append(np.power(abs(s_m-s_p),2))
spareman=1-(6*sum(R)/(n*(n^2-1)))       