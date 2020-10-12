#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston
"""
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
from utils import cc_coef


def dense_network_MTL(num_nodes):
    inputs = Input((6373,))
    encode = Dense(num_nodes)(inputs)
    encode = Dropout(0.3)(encode)
    encode = Dense(num_nodes, activation='relu')(encode)
    encode = Dropout(0.3)(encode)
    encode = Dense(num_nodes, activation='relu')(encode)
    output_act = Dense(units=1, activation='linear')(encode)
    output_dom = Dense(units=1, activation='linear')(encode)
    output_val = Dense(units=1, activation='linear')(encode)
    adam = Adam(lr=0.0001)
    model = Model(inputs=inputs, outputs=[output_act, output_dom, output_val])
    model.compile(optimizer=adam, loss=[cc_coef, cc_coef, cc_coef])
    return model

def dense_network_class(num_nodes, num_class):
    inputs = Input((6373,))
    encode = Dense(num_nodes)(inputs)
    encode = Dropout(0.3)(encode)
    encode = Dense(num_nodes, activation='relu')(encode)
    encode = Dropout(0.3)(encode)
    encode = Dense(num_nodes, activation='relu')(encode)
    outputs = Dense(units=num_class, activation='softmax')(encode)
    adam = Adam(lr=0.0001)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=adam, loss='categorical_crossentropy')
    return model
