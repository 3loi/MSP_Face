#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import numpy as np
import json
from keras import backend as K

# 5-class case
def emo_constraint_5class(arry):
    index = []
    for i in range(len(arry)):
        if (arry[i]!='C')&(arry[i]!='F')&(arry[i]!='O')&(arry[i]!='U')&(arry[i]!='X'):
            index.append(i)
    return index

# 8-class case
def emo_constraint_8class(arry):
    index = []
    for i in range(len(arry)):
        if (arry[i]!='O')&(arry[i]!='X'):
            index.append(i)
    return index

# loading data paths and labels with corresponding constraints 
def getPaths(label_type, split_set, num_class=None):
    """
    This function is for filtering data by different constraints of label
    Args:
        label_type$ (str): 'class' or 'attr'
        split_set$ (str): 'Train', 'Validation' or 'Test' are supported.
        num_class$ (str): '5-class' or '8-class' if label_type=='class'
    """
    
    path_label = '/media/winston/UTD-MSP/Speech_Datasets/MSP-Face/Labels/label_consensus.json'
    with open(path_label) as json_file:
        label_table = json.load(json_file)  
    whole_fnames = []
    emo_class = []
    emo_act = []
    emo_dom = []
    emo_val = []
    split_sets = []
    for fname in label_table.keys():
        whole_fnames.append(fname.replace('.mp4','.wav'))
        emo_class.append(label_table[fname][0]['EmoClass_Major'])
        emo_act.append(float(label_table[fname][0]['EmoAct']))
        emo_dom.append(float(label_table[fname][0]['EmoDom']))
        emo_val.append(float(label_table[fname][0]['EmoVal']))
        split_sets.append(label_table[fname][0]['Split_Set'])
    _paths = []
    _label_act = []
    _label_dom = []
    _label_val = []
    _label_class = []    
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set: 
            _paths.append(whole_fnames[i])
            _label_act.append(emo_act[i])
            _label_dom.append(emo_dom[i])
            _label_val.append(emo_val[i])
            _label_class.append(emo_class[i])
        else:
            pass
    if label_type == 'attr':
        return np.array(_paths), np.array(_label_act), np.array(_label_dom), np.array(_label_val)
    elif label_type == 'class':
        if num_class == '5-class':
            return np.array(_paths)[emo_constraint_5class(np.array(_label_class))], np.array(_label_class)[emo_constraint_5class(np.array(_label_class))]
        elif num_class == '8-class':
            return np.array(_paths)[emo_constraint_8class(np.array(_label_class))], np.array(_label_class)[emo_constraint_8class(np.array(_label_class))]
        
def class2onehot_5class(emo_class):
    if emo_class=='H':
        onehot = np.array([0,0,0,0,1])
    elif emo_class=='A':
        onehot = np.array([0,0,0,1,0])
    elif emo_class=='N':
        onehot = np.array([0,0,1,0,0])
    elif emo_class=='D':
        onehot = np.array([0,1,0,0,0])
    elif emo_class=='S':
        onehot = np.array([1,0,0,0,0])
    return onehot  

def class2onehot_8class(emo_class):
    if emo_class=='A':
        onehot = np.array([0,0,0,0,0,0,0,1])
    elif emo_class=='C':
        onehot = np.array([0,0,0,0,0,0,1,0])
    elif emo_class=='D':
        onehot = np.array([0,0,0,0,0,1,0,0])
    elif emo_class=='F':
        onehot = np.array([0,0,0,0,1,0,0,0])
    elif emo_class=='H':
        onehot = np.array([0,0,0,1,0,0,0,0])
    elif emo_class=='N':
        onehot = np.array([0,0,1,0,0,0,0,0])      
    elif emo_class=='S':
        onehot = np.array([0,1,0,0,0,0,0,0])        
    elif emo_class=='U':
        onehot = np.array([1,0,0,0,0,0,0,0])        
    return onehot        

def softprob2class_5class(pred_prob_arry):
    if np.argmax(pred_prob_arry)==4:
        emo_class = 'H'
    elif np.argmax(pred_prob_arry)==3:
        emo_class = 'A'
    elif np.argmax(pred_prob_arry)==2:
        emo_class = 'N'
    elif np.argmax(pred_prob_arry)==1:
        emo_class = 'D'        
    elif np.argmax(pred_prob_arry)==0:
        emo_class = 'S'     
    return emo_class

def softprob2class_8class(pred_prob_arry):
    if np.argmax(pred_prob_arry)==7:
        emo_class = 'A'
    elif np.argmax(pred_prob_arry)==6:
        emo_class = 'C'
    elif np.argmax(pred_prob_arry)==5:
        emo_class = 'D'
    elif np.argmax(pred_prob_arry)==4:
        emo_class = 'F'        
    elif np.argmax(pred_prob_arry)==3:
        emo_class = 'H'
    elif np.argmax(pred_prob_arry)==2:
        emo_class = 'N'               
    elif np.argmax(pred_prob_arry)==1:
        emo_class = 'S'        
    elif np.argmax(pred_prob_arry)==0:
        emo_class = 'U'            
    return emo_class

def evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value[:,0])
    ccc = 2*predicted_value[:,0].std()*true_value.std()*corr_coeff[0,1]/(predicted_value[:,0].var() + true_value.var() + (predicted_value[:,0].mean() - true_value.mean())**2)
    return(ccc,corr_coeff)

def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)                                                                                                                                                                                              
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))
