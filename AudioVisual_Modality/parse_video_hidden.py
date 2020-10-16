#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import pickle
import os
from scipy.io import savemat


# create folders for output hidden features of video part
if not os.path.isdir('./Fusion_Features/3-attribute/Videos/'):
    os.makedirs('./Fusion_Features/3-attribute/Videos/')    

if not os.path.isdir('./Fusion_Features/5-class/Videos/'):
    os.makedirs('./Fusion_Features/5-class/Videos/') 
    
if not os.path.isdir('./Fusion_Features/8-class/Videos/'):
    os.makedirs('./Fusion_Features/8-class/Videos/')


# loading video part's saving outputs
feat_all_5class = pickle.load(open( './Fusion_Features/5class_LSTM512_features.p', "rb" ))
feat_all_8class = pickle.load(open( './Fusion_Features/8class_LSTM512_features.p', "rb" ))
feat_all_attr = pickle.load(open( './Fusion_Features/VAD_LSTM512_features.p', "rb" ))


# saving features output with corresponding file name
for fname in feat_all_5class.keys():
    savemat('./Fusion_Features/5-class/Videos/'+fname+'.mat', {'Feat':feat_all_5class[fname]})
for fname in feat_all_8class.keys():
    savemat('./Fusion_Features/8-class/Videos/'+fname+'.mat', {'Feat':feat_all_8class[fname]})
for fname in feat_all_attr.keys():
    savemat('./Fusion_Features/3-attribute/Videos/'+fname+'.mat', {'Feat':feat_all_attr[fname]})

