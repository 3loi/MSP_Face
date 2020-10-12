#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import numpy as np
import os
from scipy.io import loadmat, savemat
import random
from utils import getPaths

# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)


if __name__=='__main__': 
    data_root = '/media/winston/UTD-MSP/Speech_Datasets/MSP-Face/Features/OpenSmile_func_IS13ComParE/feat_mat/'
    fnames, Train_Label_act, Train_Label_dom, Train_Label_val = getPaths(label_type='attr', split_set='Train', num_class=None)
    
    # Output normalize parameters folder based on the training set
    if not os.path.isdir('./NormTerm/'):
        os.makedirs('./NormTerm/')      
    
    # Acoustic-Feature Normalization based on Training Set
    Train_Data = []
    for i in range(len(fnames)):
        data = loadmat(data_root + fnames[i].replace('.wav','.mat'))['Audio_data']
        data = data.reshape(-1)
        Train_Data.append(data)
    Train_Data = np.array(Train_Data)

    # Feature Normalization Parameters
    Feat_mean_All = np.mean(Train_Data,axis=0)
    Feat_std_All = np.std(Train_Data,axis=0)
    savemat('./NormTerm/feat_norm_means.mat', {'normal_para':Feat_mean_All})
    savemat('./NormTerm/feat_norm_stds.mat', {'normal_para':Feat_std_All})
    Label_mean_Act = np.mean(Train_Label_act)
    Label_std_Act = np.std(Train_Label_act)
    savemat('./NormTerm/act_norm_means.mat', {'normal_para':Label_mean_Act})
    savemat('./NormTerm/act_norm_stds.mat', {'normal_para':Label_std_Act})
    Label_mean_Dom = np.mean(Train_Label_dom)
    Label_std_Dom = np.std(Train_Label_dom)    
    savemat('./NormTerm/dom_norm_means.mat', {'normal_para':Label_mean_Dom})
    savemat('./NormTerm/dom_norm_stds.mat', {'normal_para':Label_std_Dom})   
    Label_mean_Val = np.mean(Train_Label_val)
    Label_std_Val = np.std(Train_Label_val)      
    savemat('./NormTerm/val_norm_means.mat', {'normal_para':Label_mean_Val})
    savemat('./NormTerm/val_norm_stds.mat', {'normal_para':Label_std_Val})    
