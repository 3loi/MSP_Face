#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
# set the matplotlib backend so figures can be saved in the background

# import the necessary packages
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model, model_from_json

from config import build_affectnet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import AlexNet
from pyimagesearch.utils.loss import customLoss
from tqdm import tqdm
import pickle
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from glob import glob


# In[3]:


# load the RGB means for the training set
means = json.loads(open(MEANS_PATH).read())


# In[5]:


# initialize the image preprocessors
sp = SimplePreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)
pp = PatchPreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()


# In[6]:


def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)                                                                                                                                                                                             
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))


# In[ ]:


model = load_model(MODEL_PATH, custom_objects={'cc_coef':                   
cc_coef})


# In[ ]:


model_out = model.get_layer('dense').output
vgg_feat_model = Model(inputs =model.input, outputs = model_out)


# In[ ]:



all_vids = glob(source_path)


# In[1]:


def load_img(path):
    image = cv2.imread(path)
    image2 = sp.preprocess(image)
    image2 = mp.preprocess(image2)
    image2 = iap.preprocess(image2)
    return image2

def batch_generator(frames, batch_size, width, height, channels):
    images = np.zeros((batch_size, width, height, channels))
    curr = 0
    for i, frame in enumerate(frames):
        images[curr] = load_img(frame)
        
def load_videos(path):
    all_vids = glob(source_path)
    print("videos to load: ", len(all_vids))
    for vid in all_vids:
        frames = sorted(glob(os.path.join(vid, '*.jpg')))
        imgs, names = [], []
        for frame in frames:
            base = os.path.basename(frame)
            img = load_img(frame)
            imgs.append(img)
            names.append(base)
        yield vid, names, np.array(imgs)


# In[13]:


for vid_path, frames, imgs in tqdm(load_videos(path)):
    pred = vgg_feat_model.predict(imgs)
    
    data = {
        'feat':pred,
        'frames':frames
    }
    dest = os.path.join(FEAT_PATH, vid_path.split(os.path.sep)[-2])
    if not os.path.exists(dest):
        os.makedirs(dest)
    vid_name = os.path.basename(vid_path)
    save_name = os.path.join(dest, vid_name + '.p')
    pickle.dump( data, open( save_name, "wb" ) )


# In[ ]:




