#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winstonlin
"""
import matplotlib.pyplot as plt
import os
from keras.callbacks import ModelCheckpoint
import random
from dataloader import DataGenerator
from model import dense_network_MTL, dense_network_class
import argparse
# Ignore warnings & Fix random seed
import warnings
warnings.filterwarnings("ignore")
random.seed(999)
random_seed=99


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

params_train = {'batch_size': batch_size,
                'label_type': label_type,
                'num_class': num_class,
                'split_set': 'Train',
                'shuffle': True}

params_valid = {'batch_size': 200,
                'label_type': label_type,
                'num_class': num_class,
                'split_set': 'Validation',
                'shuffle': False}

# Output models saving folder
if not os.path.isdir('./Models/'):
    os.makedirs('./Models/')  

# Generators
training_generator = DataGenerator(root_dir, **params_train)
validation_generator = DataGenerator(root_dir, **params_valid)

# Model Settings
if label_type == 'attr':
    filepath='./Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.hdf5'
elif label_type == 'class':
    filepath='./Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Model structure loading
if label_type == 'attr':
    model = dense_network_MTL(num_nodes=num_nodes)
elif label_type == 'class':
    model = dense_network_class(num_nodes=num_nodes, num_class=int(num_class.split('-')[0]))

# Model Training on Batch
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=12,
                    epochs=epochs, 
                    verbose=1,
                    callbacks=callbacks_list)

# Show training & validation loss
v_loss = model.history.history['val_loss']
t_loss = model.history.history['loss']
plt.plot(t_loss,'b')
plt.plot(v_loss,'r')
if label_type == 'attr':
    plt.savefig('./Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+label_type+'.png')
elif label_type == 'class':
    plt.savefig('./Models/DenseNN_model[epoch'+str(epochs)+'-batch'+str(batch_size)+'-nodes'+str(num_nodes)+']_'+num_class+'.png')