#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from scipy.io import loadmat
from sklearn.metrics import f1_score
from utils import getPaths, cc_coef, evaluation_metrics
from utils import softprob2class_5class, softprob2class_8class



def fusion_network_MTL(num_nodes):
    inputs = Input((768,))
    encode = Dense(num_nodes, activation='relu')(inputs)
    encode = Dense(num_nodes, activation='relu')(encode)
    output_act = Dense(units=1, activation='linear')(encode)
    output_dom = Dense(units=1, activation='linear')(encode)
    output_val = Dense(units=1, activation='linear')(encode)
    adam = Adam(lr=0.0001)
    model = Model(inputs=inputs, outputs=[output_act, output_dom, output_val])
    model.compile(optimizer=adam, loss=[cc_coef, cc_coef, cc_coef])
    return model    

def fusion_network_class(num_nodes, num_class):
    inputs = Input((768,))
    encode = Dense(num_nodes, activation='relu')(inputs)
    encode = Dense(num_nodes, activation='relu')(encode)    
    outputs = Dense(units=num_class, activation='softmax')(encode)
    adam = Adam(lr=0.0001)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model
###############################################################################


# Parameters
batch_size = 256
num_nodes = 256
epochs = 50

label_type = 'attr'
#label_type = 'class'

num_class = None
#num_class = '5-class'
#num_class = '8-class'

# Hidden Features Paths Setting
if label_type == 'attr':
    root_dir = './Fusion_Features/3-attribute'
elif label_type == 'class':
    if num_class == '5-class':
        root_dir = './Fusion_Features/5-class'
    elif num_class == '8-class':
        root_dir = './Fusion_Features/8-class'

# Loading Paths & Labels
if label_type == 'class':
    paths_test, labels_class_test = getPaths(label_type, split_set='Test', num_class=num_class)
elif label_type == 'attr':
    # Loading Norm-Label
    Label_mean_act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std_act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
    Label_mean_dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std_dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
    Label_mean_val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std_val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]     
    paths_test, labels_act_test, labels_dom_test, labels_val_test = getPaths(label_type, split_set='Test', num_class=num_class)

# Loading Hidden Features (Testing set)
X_Test = []
Y_Test_Class = []
Y_Test_Act = []
Y_Test_Dom = []
Y_Test_Val = []
for i in range(len(paths_test)):
    try: # deal with missing files
        x_audio = loadmat(root_dir + '/Audios/' + paths_test[i].replace('.wav','.mat'))['Feat']
        x_video = loadmat(root_dir + '/Videos/' + paths_test[i].replace('.wav','.mat'))['Feat']
        # fusing audio-visual hidden features
        x = np.concatenate((x_audio, x_video),axis=1)
        x = x.reshape(-1) 
        X_Test.append(x)
        if label_type == 'class':
            y = labels_class_test[i]
            Y_Test_Class.append(y)
        elif label_type == 'attr':
            y_act = labels_act_test[i]
            y_dom = labels_dom_test[i]
            y_val = labels_val_test[i]            
            Y_Test_Act.append(y_act)
            Y_Test_Dom.append(y_dom)
            Y_Test_Val.append(y_val)               
    except:
        pass
    
if label_type == 'class':
    X_Test = np.array(X_Test)
    Y_Test_Class = np.array(Y_Test_Class)
elif label_type == 'attr':
    X_Test = np.array(X_Test)
    Y_Test_Act = np.array(Y_Test_Act)    
    Y_Test_Dom = np.array(Y_Test_Dom)
    Y_Test_Val = np.array(Y_Test_Val)

# Loading Models
if label_type == 'attr':
    filepath='./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.hdf5'
elif label_type == 'class':
    filepath='./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.hdf5'

# Testing process
if label_type == 'class':
    best_model = fusion_network_class(num_nodes=num_nodes, num_class=int(num_class.split('-')[0]))
    best_model.load_weights(filepath)
    pred_class_prob = best_model.predict(X_Test)
    # class prob => class label
    pred_class = []
    for i in range(len(pred_class_prob)):
        if num_class == '5-class':
            pred_class.append(softprob2class_5class(pred_class_prob[i,:]))
        elif num_class == '8-class':
            pred_class.append(softprob2class_8class(pred_class_prob[i,:]))        
    pred_class = np.array(pred_class)
    # compute evaluation metrics
    fs_test_uar = f1_score(Y_Test_Class, pred_class, average='macro')
    fs_test_total = f1_score(Y_Test_Class, pred_class, average='micro')
    print('Test F1-Score(UAR): '+str(fs_test_uar))
    print('Test F1-Score(Total): '+str(fs_test_total))

elif label_type == 'attr':
    best_model = fusion_network_MTL(num_nodes=num_nodes)
    best_model.load_weights(filepath)
    pred_act, pred_dom, pred_val = best_model.predict(X_Test)
    # de-normalization
    pred_act = (Label_std_act*pred_act)+Label_mean_act
    pred_dom = (Label_std_dom*pred_dom)+Label_mean_dom
    pred_val = (Label_std_val*pred_val)+Label_mean_val
    # Output Predict Reulst
    pred_Rsl_Act = str(evaluation_metrics(Y_Test_Act, pred_act)[0])
    pred_Rsl_Dom = str(evaluation_metrics(Y_Test_Dom, pred_dom)[0])
    pred_Rsl_Val = str(evaluation_metrics(Y_Test_Val, pred_val)[0])
    print('Act-CCC: '+str(pred_Rsl_Act))
    print('Dom-CCC: '+str(pred_Rsl_Dom))
    print('Val-CCC: '+str(pred_Rsl_Val))
