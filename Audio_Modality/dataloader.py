#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import numpy as np
from scipy.io import loadmat
import keras
import random
from utils import getPaths, class2onehot_5class, class2onehot_8class
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, root_dir, batch_size, label_type, num_class, split_set, shuffle=True):
        'Initialization'
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.label_type = label_type                      # 'attr' or 'class'
        self.num_class = num_class                        # '5-class' or '8-class' if label_type=='class'
        self.split_set = split_set                        # 'Train' or 'Validation'
        self.shuffle = shuffle
        self.on_epoch_end()
        
        # Loading Norm-Feature
        self.Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label
        self.Label_mean_act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
        self.Label_std_act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
        self.Label_std_dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
        self.Label_mean_val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
        self.Label_std_val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]             

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(getPaths(self.label_type, self.split_set, self.num_class)[0])/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Loading Paths & Labels
        if self.label_type == 'attr':
            _paths, _labels_act, _labels_dom, _labels_val = getPaths(self.label_type, self.split_set, self.num_class)
            # Find Batch list of Loading Paths
            list_paths_temp = [_paths[k] for k in indexes]
            list_act_temp = [_labels_act[k] for k in indexes]
            list_dom_temp = [_labels_dom[k] for k in indexes]
            list_val_temp = [_labels_val[k] for k in indexes]
            # Generate data
            data, label = self.__data_generation_attr(list_paths_temp, list_act_temp, list_dom_temp, list_val_temp)            
            
        elif self.label_type == 'class':
            _paths, _labels_class = getPaths(self.label_type, self.split_set, self.num_class)
            # Find Batch list of Loading Paths
            list_paths_temp = [_paths[k] for k in indexes]
            list_class_temp = [_labels_class[k] for k in indexes]
            # Generate data
            data, label = self.__data_generation_class(list_paths_temp, list_class_temp)
        return data, label        

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.label_type == 'attr':
            _paths, _, _, _ = getPaths(self.label_type, self.split_set, self.num_class)
        elif self.label_type == 'class':
            _paths, _ = getPaths(self.label_type, self.split_set, self.num_class)
            
        self.indexes = np.arange(len(_paths))
        if self.shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(self.indexes)

    def __data_generation_attr(self, list_paths_temp, list_act_temp, list_dom_temp, list_val_temp):
        'Generates data containing batch_size samples' # X : (n_samples, feat_dim)
        # Initialization
        X = np.empty((self.batch_size, 6373))
        y_act = np.empty((self.batch_size))
        y_dom = np.empty((self.batch_size))
        y_val = np.empty((self.batch_size))
        # Generate data
        for i in range(len(list_paths_temp)):
            # Store Norm-Data
            x = loadmat(self.root_dir + list_paths_temp[i].replace('.wav','.mat'))['Audio_data']
            x = (x-self.Feat_mean_All)/self.Feat_std_All    # Feature Normalization
            x = x.reshape(-1)
            # Bounded NormFeat Range -3~3 and assign NaN to 0
            x[np.isnan(x)]=0
            x[x>3]=3
            x[x<-3]=-3            
            X[i] = x
            # Store Norm-Label
            y_act[i] = (list_act_temp[i]-self.Label_mean_act)/self.Label_std_act
            y_dom[i] = (list_dom_temp[i]-self.Label_mean_dom)/self.Label_std_dom
            y_val[i] = (list_val_temp[i]-self.Label_mean_val)/self.Label_std_val
        return (X, [y_act, y_dom, y_val])

    def __data_generation_class(self, list_paths_temp, list_class_temp):
        'Generates data containing batch_size samples' # X : (n_samples, feat_dim)
        # Initialization
        X = np.empty((self.batch_size, 6373))
        if self.num_class == '5-class':
            y = np.empty((self.batch_size, 5))  # 5-class classification
        elif self.num_class == '8-class':
            y = np.empty((self.batch_size, 8))  # 8-class classification
            
        # Generate data
        for i in range(len(list_paths_temp)):
            # Store Norm-Data
            x = loadmat(self.root_dir + list_paths_temp[i].replace('.wav','.mat'))['Audio_data']
            x = (x-self.Feat_mean_All)/self.Feat_std_All    # Feature Normalization
            x = x.reshape(-1)
            # Bounded NormFeat Range -3~3 and assign NaN to 0
            x[np.isnan(x)]=0
            x[x>3]=3
            x[x<-3]=-3            
            X[i] = x
            # Store Label
            if self.num_class == '5-class':
                y[i] = class2onehot_5class(list_class_temp[i])
            elif self.num_class == '8-class':
                y[i] = class2onehot_8class(list_class_temp[i])               
        return (X ,y)
