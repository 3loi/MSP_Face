#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from scipy.io import loadmat
from utils import getPaths, cc_coef
from utils import class2onehot_5class, class2onehot_8class 
import argparse



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



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epoch", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_type", required=True)
argparse.add_argument("-nodes", "--num_nodes", required=True)
argparse.add_argument("-nc", "--num_class")
args = vars(argparse.parse_args())

# Parameters
shuffle = True
random_seed = 99
batch_size = int(args['batch_size'])
epochs = int(args['epoch'])
num_nodes = int(args['num_nodes'])
label_type = args['emo_type']
try:
    num_class = args['num_class']
except:
    pass

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
    paths_valid, labels_class_valid = getPaths(label_type, split_set='Validation', num_class=num_class)
    paths_train, labels_class_train = getPaths(label_type, split_set='Train', num_class=num_class)
elif label_type == 'attr':
    # Loading Norm-Label
    Label_mean_act = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std_act = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
    Label_mean_dom = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std_dom = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
    Label_mean_val = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std_val = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]     
    paths_valid, labels_act_valid, labels_dom_valid, labels_val_valid = getPaths(label_type, split_set='Validation', num_class=num_class)
    paths_train, labels_act_train, labels_dom_train, labels_val_train = getPaths(label_type, split_set='Train', num_class=num_class)    

# shuffle the training set
indexes = np.arange(len(paths_train))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indexes)
    
if label_type == 'class':
    shuffle_paths_train = [paths_train[k] for k in indexes]
    shuffle_class_train = [labels_class_train[k] for k in indexes]
elif label_type == 'attr':
    shuffle_paths_train = [paths_train[k] for k in indexes]
    shuffle_act_train = [labels_act_train[k] for k in indexes]
    shuffle_dom_train = [labels_dom_train[k] for k in indexes]
    shuffle_val_train = [labels_val_train[k] for k in indexes]

# Loading Hidden Features (Training set)
X_Train = []
Y_Train_Class = []
Y_Train_Act = []
Y_Train_Dom = []
Y_Train_Val = []
for i in range(len(shuffle_paths_train)):
    try: # deal with missing files
        x_audio = loadmat(root_dir + '/Audios/' + shuffle_paths_train[i].replace('.wav','.mat'))['Feat']
        x_video = loadmat(root_dir + '/Videos/' + shuffle_paths_train[i].replace('.wav','.mat'))['Feat']
        # fusing audio-visual hidden features
        x = np.concatenate((x_audio, x_video),axis=1)
        x = x.reshape(-1) 
        X_Train.append(x)
        if label_type == 'class':     # STL
            # class to one-hot label
            if num_class == '5-class':
                y = class2onehot_5class(shuffle_class_train[i])
            elif num_class == '8-class':
                y = class2onehot_8class(shuffle_class_train[i])
            Y_Train_Class.append(y)
            
        elif label_type == 'attr':    # MTL  
            # normalize regression label
            y_act = (shuffle_act_train[i]-Label_mean_act)/Label_std_act
            y_dom = (shuffle_dom_train[i]-Label_mean_dom)/Label_std_dom
            y_val = (shuffle_val_train[i]-Label_mean_val)/Label_std_val            
            Y_Train_Act.append(y_act)
            Y_Train_Dom.append(y_dom)
            Y_Train_Val.append(y_val)
    except:
        pass

if label_type == 'class':
    X_Train = np.array(X_Train)
    Y_Train_Class = np.array(Y_Train_Class)
elif label_type == 'attr':
    X_Train = np.array(X_Train)
    Y_Train_Act = np.array(Y_Train_Act)    
    Y_Train_Dom = np.array(Y_Train_Dom)
    Y_Train_Val = np.array(Y_Train_Val)

# Loading Hidden Features (Validation set)
X_Valid = []
Y_Valid_Class = []
Y_Valid_Act = []
Y_Valid_Dom = []
Y_Valid_Val = []
for i in range(len(paths_valid)):
    try: # deal with missing files
        x_audio = loadmat(root_dir + '/Audios/' + paths_valid[i].replace('.wav','.mat'))['Feat']
        x_video = loadmat(root_dir + '/Videos/' + paths_valid[i].replace('.wav','.mat'))['Feat']
        # fusing audio-visual hidden features
        x = np.concatenate((x_audio, x_video),axis=1)
        x = x.reshape(-1)
        X_Valid.append(x)
        if label_type == 'class':
            # class to one-hot label
            if num_class == '5-class':
                y = class2onehot_5class(labels_class_valid[i])
            elif num_class == '8-class':
                y = class2onehot_8class(labels_class_valid[i])
            Y_Valid_Class.append(y)
        elif label_type == 'attr':     
            y_act = (labels_act_valid[i]-Label_mean_act)/Label_std_act
            y_dom = (labels_dom_valid[i]-Label_mean_dom)/Label_std_dom
            y_val = (labels_val_valid[i]-Label_mean_val)/Label_std_val            
            Y_Valid_Act.append(y_act)
            Y_Valid_Dom.append(y_dom)
            Y_Valid_Val.append(y_val)        
    except:
        pass
    
if label_type == 'class':
    X_Valid = np.array(X_Valid)
    Y_Valid_Class = np.array(Y_Valid_Class)
elif label_type == 'attr':
    X_Valid = np.array(X_Valid)
    Y_Valid_Act = np.array(Y_Valid_Act)    
    Y_Valid_Dom = np.array(Y_Valid_Dom)
    Y_Valid_Val = np.array(Y_Valid_Val)

# loading model structure
if label_type == 'class':
    model = fusion_network_class(num_nodes=num_nodes, num_class=int(num_class.split('-')[0]))
elif label_type == 'attr':
    model = fusion_network_MTL(num_nodes=num_nodes)
#print(model.summary()) 

# Output fusion models saving folder
if not os.path.isdir('./Fusion_Models/'):
    os.makedirs('./Fusion_Models/')  

# setting model checkpoints
if label_type == 'attr':
    filepath='./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.hdf5'
elif label_type == 'class':
    filepath='./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# model fitting
if label_type == 'class':
    model.fit(x=X_Train, 
              y=Y_Train_Class, 
              batch_size=batch_size, 
              epochs=epochs,
              validation_data=(X_Valid, Y_Valid_Class),
              verbose=1,
              callbacks=callbacks_list)
    
elif label_type == 'attr':
    model.fit(x=X_Train, 
              y=([Y_Train_Act, Y_Train_Dom, Y_Train_Val]), 
              batch_size=batch_size, 
              epochs=epochs,
              validation_data=(X_Valid, [Y_Valid_Act, Y_Valid_Dom, Y_Valid_Val]),
              verbose=1,
              callbacks=callbacks_list)

# Show training & validation loss
v_loss = model.history.history['val_loss']
t_loss = model.history.history['loss']
plt.plot(t_loss,'b')
plt.plot(v_loss,'r')
if label_type == 'attr':
    plt.savefig('./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.png')
elif label_type == 'class':
    plt.savefig('./Fusion_Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.png')
