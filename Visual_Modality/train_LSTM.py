#!/usr/bin/env python
# coding: utf-8


import json
import pandas as pd
import numpy as np
import os
import pickle
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Convolution2D,Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import TimeDistributed, LSTM, CuDNNGRU, CuDNNLSTM, RNN, Masking, Bidirectional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adadelta
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support as score

from config import *




config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)





def get_msp_face(video_path, file_path, mode = None):
    
    name2index = {}

    lines = open(name_path, 'r').readlines()
    for i, class_name in enumerate(lines):
        class_name = class_name.split()[1]
        name2index[str(class_name)[0]]=i
    with open(file_path) as f:
        data = json.load(f)
    
    video_files = []
    label_files = []
    valence = []
    arousal = []
    dominance = []
    for file in data:
        buf = data[file][0]
        #modes can be Train/Test/Validation
        if mode != None and buf['Split_Set'].lower() != mode.lower():
            continue
        emo = buf['EmoClass_Major']


        pathname, _ = os.path.splitext(file)
        #print(pathname, file)
        video_files.append(os.path.join(video_path, emo, pathname))
        valence.append(float(buf['EmoVal']))
        arousal.append(float(buf['EmoAct']))
        dominance.append(float(buf['EmoDom']))
        label_files.append(name2index[emo])
    if NUM_CLASSES = 'VAD':
        return video_files,  np.array(valence), np.array(arousal), np.array(dominance)
    else:
        return video_files, np.array(label_files)
    
def segment_vids(vids, labels):
    step = 30
    clip_len = 60
    labs = []
    segments = []
    for vid, lab in zip(vids, labels):
        n_frames = len(vid)

        for i in range(0, n_frames, step):
            if i+clip_len >= n_frames:
                continue
            end =  i + clip_len
            segments.append(vid[i:end])
            labs.append(lab)
    return np.array(segments), labs

def load_vids(paths, labels):
    data = []
    train_labels = []
    for i, vid in enumerate(paths):
        try:
            data.append(pickle.load( open( vid + '.p', "rb" ) )['feat'])
            train_labels.append(labels[i])
        except:
            print("Unable to find file: ", vid)
    return data, train_labels

def load_vids_vad(paths, v, a, d):
    data = []
    train_labels = []
    val, act, dom = [], [], []
    for i, vid in enumerate(paths):
        try:
            data.append(pickle.load( open( vid + '.p', "rb" ) )['feat'])

            val.append(v[i])
            act.append(a[i])
            dom.append(d[i])
        except:
            print("Unable to find file: ", vid)
    return data, np.array(val), np.array(act), np.array(dom)

def cc_evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value[:,0])
    ccc = 2*predicted_value[:,0].std()*true_value.std()*corr_coeff[0,1]/(predicted_value[:,0].var() + true_value.var() + (predicted_value[:,0].mean() - true_value.mean())**2)
    return(ccc,corr_coeff)   

def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)                                                                                                                                                                                             
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))







if NUM_CLASSES == 'VAD':
    train_vids, val_train, act_train, dom_train = get_msp_face(FEAT_PATH, LABEL_PATH, mode = 'train', NAME_PATH)
    train_vids, val_train, act_train, dom_train = shuffle(train_vids, val_train, act_train, dom_train)
    train_vids, val_train, act_train, dom_train = load_vids_vad(train_vids , val_train, act_train, dom_train)
    
    valid_vids, val_valid, act_valid, dom_valid = get_msp_face(FEAT_PATH, LABEL_PATH, mode = 'validation', NAME_PATH)
    valid_vids, val_valid, act_valid, dom_valid = load_vids_vad(valid_vids, val_valid, act_valid, dom_valid)
    
    valid_vids, val_valid, act_valid, dom_valid = get_msp_face(FEAT_PATH, LABEL_PATH, mode = 'test', NAME_PATH)
    valid_vids, val_valid, act_valid, dom_valid = load_vids_vad(valid_vids, val_valid, act_valid, dom_valid)
else:
    train_vids, train_labels = get_msp_face(FEAT_PATH, LABEL_PATH, NAME_PATH, mode = 'train', num_classes = NUM_CLASSES)
    train_vids, train_labels = load_vids(train_vids, train_labels)
    train_labels = np.eye(NUM_CLASSES)[train_labels]
    train_vids, train_labels = shuffle(train_vids, train_labels)

    valid_vids, valid_labels = get_msp_face(FEAT_PATH, LABEL_PATH, NAME_PATH, mode = 'validation', num_classes = NUM_CLASSES)
    valid_vids, valid_labels = load_vids(valid_vids, valid_labels)
    valid_labels = np.eye(NUM_CLASSES)[valid_labels]

    test_vids, test_labels = get_msp_face(FEAT_PATH, LABEL_PATH, NAME_PATH,, mode = 'test' num_classes = NUM_CLASSES)
    test_vids, test_labels = load_vids(test_vids, test_labels)
    test_labels = np.eye(NUM_CLASSES)[test_labels]



dropout = 0.5
activation = 'relu'
m_input = Input(shape=(None,1024), name = 'Input_mesh')
model = m_input
model = tf.keras.layers.Masking(mask_value=0.0)(model)
model = TimeDistributed(Dense(512, activation = activation))(model)
model = TimeDistributed(BatchNormalization())(model)
model = TimeDistributed(Dropout(dropout))(model)

model = TimeDistributed(Dense(256, activation = activation))(model)
model = TimeDistributed(BatchNormalization())(model)
model = TimeDistributed(Dropout(dropout))(model)

model = TimeDistributed(Dense(128, activation = activation))(model)
model = TimeDistributed(BatchNormalization())(model)
model = TimeDistributed(Dropout(dropout))(model)


model = LSTM(512, dropout=dropout, recurrent_dropout=dropout, return_sequences=False)(model)
V = Dense(1)(model)
A = Dense(1)(model)
D = Dense(1)(model)

model = Model(inputs=m_input, outputs = [V,A,D])

ada = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss=cc_coef, optimizer='adam',
	metrics=[cc_coef])



train_vids  = tf.keras.preprocessing.sequence.pad_sequences(
    train_vids, padding="pre"
)



valid_vids  = tf.keras.preprocessing.sequence.pad_sequences(
    valid_vids, padding="pre"
)




history = model.fit(train_vids, [val_train, act_train, dom_train],
         validation_data=(valid_vids, [val_valid, act_valid, dom_valid]),
         batch_size=32,
         epochs = 100,
         verbose = 1,
         )



test_vids, test_labels = get_msp_face(video_path, label_path, name_path)
test_vids = load_vids(test_vids)




if NUM_CLASSES == 'VAD':
    val_pred, act_pred, dom_pred =model.predict(test_vids)
    print('VAl evaluation: ',cc_evaluation_metrics(val_test, val_pred))
    print('ACT evaluation: ',cc_evaluation_metrics(act_test, act_pred))
    print('DOm evaluation: ',cc_evaluation_metrics(dom_test, dom_pred))
else:
    y_pred=model.predict(test_vids)
    y_test = test_labels.argmax(axis=1)
    
    
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1), labels = [num for num in range(int('8'))])

    f1 = f1_score(y_test, y_pred.argmax(axis=1), average = 'micro')
    print(f1)
    print(f1_score(y_test, y_pred.argmax(axis=1), average=None))
    print(precision_score(y_test, y_pred.argmax(axis=1), average=None))

    print(recall_score(y_test, y_pred.argmax(axis=1), average=None))  


    precision, recall, fscore, support = score(y_test, y_pred.argmax(axis=1))

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
    print('precision: {}'.format(np.mean(precision)))
    print('recall: {}'.format(np.mean(recall)))
    print('fscore: {}'.format(np.mean(fscore)))
    print('support: {}'.format(np.mean(support)))



print("Saving model...")
model.save(MODEL_PATH, overwrite=True)




model = load_model(MODEL_PATH, custom_objects={'cc_coef': cc_coef})
model.summary()



model_out = model.get_layer('lstm').output
feat_model = Model(inputs =model.input, outputs = model_out)


feat_model.summary()



all_vids_name, _ ,_ ,_ = get_msp_face(video_path, label_path)
all_vids, _, _, _ = load_vids(all_vids_name, a, b, c )



all_vids1  = tf.keras.preprocessing.sequence.pad_sequences(
    all_vids, padding="pre"
)



preds = feat_model.predict(all_vids1)






features = {}


for pre, name in tqdm(zip(preds, all_vids_name)):
    features[name.split(os.path.sep)[-1]] = pre
  
    


pickle.dump( features, open( "8CLASS_LSTM512_features_NEW.p", "wb" ) )



