#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
from utils import getPaths, evaluation_metrics, softprob2class_5class, softprob2class_8class
from scipy.io import loadmat, savemat
import numpy as np
import os
from keras import backend as K
from model import dense_network_MTL, dense_network_class
from sklearn.metrics import f1_score
import argparse


argparse = argparse.ArgumentParser()
argparse.add_argument("-root", "--root_dir", required=True)
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_type", required=True)
argparse.add_argument("-nodes", "--num_nodes", required=True)
argparse.add_argument("-nc", "--num_class")
args = vars(argparse.parse_args())

# Parameters
root_dir = args['root_dir'] # e.g., XXX/Dataset/MSP-Face/Features/OpenSmile_func_IS13ComParE/feat_mat/
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
num_nodes = int(args['num_nodes'])
label_type = args['emo_type']
try:
    num_class = args['num_class']
except:
    pass

Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
Label_mean_act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
Label_std_act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
Label_mean_dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
Label_std_dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
Label_mean_val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
Label_std_val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]

# Testing Task
if label_type == 'attr':
    test_file_path, test_file_tar_act,  test_file_tar_dom, test_file_tar_val = getPaths(label_type, 'Test', num_class)
elif label_type == 'class':
    test_file_path, test_file_tar_class = getPaths(label_type, 'Test', num_class)

# Testing Data & Label
Test_Data = []
Test_Label_Act = []
Test_Label_Dom = []
Test_Label_Val = []
Test_Label_Class = []
for i in range(len(test_file_path)):
    data = loadmat(root_dir + test_file_path[i].replace('.wav','.mat'))['Audio_data']
    data = (data-Feat_mean_All)/Feat_std_All    # Feature Normalization
    data = data.reshape(-1)  
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3
    Test_Data.append(data)
    if label_type == 'attr': 
        Test_Label_Act.append(test_file_tar_act[i])
        Test_Label_Dom.append(test_file_tar_dom[i])
        Test_Label_Val.append(test_file_tar_val[i])
    elif label_type == 'class':
        Test_Label_Class.append(test_file_tar_class[i])
Test_Data = np.array(Test_Data)
Test_Label_Act = np.array(Test_Label_Act)
Test_Label_Dom = np.array(Test_Label_Dom)
Test_Label_Val = np.array(Test_Label_Val)
Test_Label_Class = np.array(Test_Label_Class)

# Regression Task => Prediction & De-Normalize Target
if label_type == 'attr':
    model_path = './Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.hdf5'
    model = dense_network_MTL(num_nodes=num_nodes)
    model.load_weights(model_path)
    pred_Label_act, pred_Label_dom, pred_Label_val = model.predict(Test_Data)
    # de-norm predictions
    pred_Label_act = (Label_std_act*pred_Label_act)+Label_mean_act
    pred_Label_dom = (Label_std_dom*pred_Label_dom)+Label_mean_dom
    pred_Label_val = (Label_std_val*pred_Label_val)+Label_mean_val
    # Output Predict Reulst
    pred_Rsl_Act = str(evaluation_metrics(Test_Label_Act, pred_Label_act)[0])
    pred_Rsl_Dom = str(evaluation_metrics(Test_Label_Dom, pred_Label_dom)[0])
    pred_Rsl_Val = str(evaluation_metrics(Test_Label_Val, pred_Label_val)[0])
    print('Act-CCC: '+str(pred_Rsl_Act))
    print('Dom-CCC: '+str(pred_Rsl_Dom))
    print('Val-CCC: '+str(pred_Rsl_Val))

# Classification Task
elif label_type == 'class':
    model_path = './Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.hdf5'
    model = dense_network_class(num_nodes=num_nodes, num_class=int(num_class.split('-')[0]))
    model.load_weights(model_path)
    pred_Label_prob = model.predict(Test_Data)
    # softmax to predict class
    pred_Label_class = []
    for i in range(len(pred_Label_prob)):
        if num_class == '5-class':
            pred_Label_class.append(softprob2class_5class(pred_Label_prob[i,:]))
        elif num_class == '8-class':
            pred_Label_class.append(softprob2class_8class(pred_Label_prob[i,:]))            
    pred_Label_class = np.array(pred_Label_class)
    # Output Predict Reulst
    fs_test_uar = f1_score(Test_Label_Class, pred_Label_class, average='macro')
    fs_test_total = f1_score(Test_Label_Class, pred_Label_class, average='micro')    
    print('Test F1-Score(UAR): '+str(fs_test_uar))
    print('Test F1-Score(Total): '+str(fs_test_total))


## Output hidden feature for AudioVisual Fusion Model
#if not os.path.isdir('./Fusion_Features/3-attribute/Audios/'):
#    os.makedirs('./Fusion_Features/3-attribute/Audios/')    
#
#if not os.path.isdir('./Fusion_Features/5-class/Audios/'):
#    os.makedirs('./Fusion_Features/5-class/Audios/') 
#    
#if not os.path.isdir('./Fusion_Features/8-class/Audios/'):
#    os.makedirs('./Fusion_Features/8-class/Audios/')
#
## generate hidden features
#if label_type == 'attr':
#    last_hidden_model = K.function([model.layers[0].input,K.learning_phase()],
#                                    [model.layers[-4].output])  
#    hidden_output = last_hidden_model([Test_Data])[0]
#    for i in range(len(test_file_path)):
#        savemat('./Fusion_Features/3-attribute/Audios/'+test_file_path[i].replace('.wav','.mat'), {'Feat':hidden_output[i]})
#
#
#elif label_type == 'class':    
#    last_hidden_model = K.function([model.layers[0].input,K.learning_phase()],
#                                [model.layers[-2].output])
#    hidden_output = last_hidden_model([Test_Data])[0]
#    for i in range(len(test_file_path)):
#        if num_class == '5-class':
#            savemat('./Fusion_Features/5-class/Audios/'+test_file_path[i].replace('.wav','.mat'), {'Feat':hidden_output[i]})
#        elif num_class == '8-class':
#            savemat('./Fusion_Features/8-class/Audios/'+test_file_path[i].replace('.wav','.mat'), {'Feat':hidden_output[i]})