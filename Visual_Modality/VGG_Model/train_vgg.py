#!/usr/bin/env python
# coding: utf-8

from config_file import *
#resrouces used from pyimagesearch for dataloading and processing
import tensorflow.keras.backend as K
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import PatchPreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import applications
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Dense
import json
import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import f1_score


# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.05,
	width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
	horizontal_flip=True, fill_mode="nearest")





# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())




# initialize the image preprocessors
sp = SimplePreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)
pp = PatchPreprocessor(IMAGE_WIDTH, IMAGE_HEIGHT)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()


# initialize the training and validation dataset generators
if NUM_CLASSES:
    trainGen = HDF5DatasetGenerator(TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
        preprocessors=[sp, mp, iap], classes=NUM_CLASSES)
    valGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	    preprocessors=[sp, mp, iap], classes=NUM_CLASSES)
else:
    trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug,
        preprocessors=[sp, mp, iap], classes=8)
    valGen = HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
	    preprocessors=[sp, mp, iap], classes=8)





# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)

#Use Imagenet or VGG-Face pre-weights
vgg = applications.VGG19(weights = "imagenet", include_top = False, 
    input_shape=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))


#Pick which layers to freeze/train
for layer in vgg.layers[:15]:
    layer.trainable = False
for layer in vgg.layers[15:]:
    layer.trainable = True

##cc loss if using V/A/D and not discrete classes
def cc_coef(y_true, y_pred):
    mu_y_true = K.mean(y_true)
    mu_y_pred = K.mean(y_pred)                                                                                                                                                                                             
    return 1 - 2 * K.mean((y_true - mu_y_true) * (y_pred - mu_y_pred)) / (K.var(y_true) + K.var(y_pred) + K.mean(K.square(mu_y_pred - mu_y_true)))

#Add FC layer then SoftMax/linear to the pre-trained weights loaded 
x = vgg.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

if num_class:
    predictions = Dense(config.NUM_CLASSES, activation="softmax")(x)
    # creating the final model 
    model = Model(inputs = vgg.input, outputs = predictions)
        
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

else:
    val = Dense(1)(x)
    act = Dense(1)(x)
    # creating the final model 
    model = Model(inputs = vgg.input, outputs = [val, act])
    model.compile(loss=cc_coef, optimizer=opt)



# train the network
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.000001)


save_model = ModelCheckpoint(
    filepath = config.MODEL_PATH,
    verbose = 1,
    period = 1,
)


#class weights if training weighted
if use_class_weights:
    if NUM_CLASSES is 5:
        class_weight = {
            0: 1.,
            1: 1.,
            2: 1.,
            3: 1.,
            4: 6.5,
            }
    elif NUM_CLASSES is 8:
        class_weight = {
            0: 1.,
            1: 1.,
            2: 1.,
            3: 1.,
            4: 6.5,
            5: 6.5,
            6: 1,
            7: 6.5,
            }
else:
    if NUM_CLASSES is 5:
        class_weight = {i:1. for i in range(5)}
    elif NUM_CLASSES is 8:
        class_weight = {i:1. for i in range(8)}


path = os.path.sep.join([OUTPUT_PATH, "{}.png".format(
	os.getpid())])

callbacks = [learning_rate_reduction, save_model]
#callbacks = [TrainingMonitor(path), learning_rate_reduction, save_model]

if num_class:
    model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages / config.BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages / config.BATCH_SIZE,
        epochs= config.EPOCHS,
        max_queue_size=10,
        class_weight = class_weight,
        callbacks=callbacks,
        verbose=1)
else:
    model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages / config.BATCH_SIZE,
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages / config.BATCH_SIZE,
        epochs= config.EPOCHS,
        max_queue_size=10,
        callbacks=callbacks,
        verbose=1)


# save the model
print("[INFO] serializing model...")
model.save(MODEL_PATH, overwrite=True)

# close the HDF5 training datasets
trainGen.close()
valGen.close()




# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops)...")
testGen = HDF5DatasetGenerator(TEST_HDF5, 64,
	preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
predictions = model.predict_generator(testGen.generator(passes=1),
	steps=testGen.numImages / 64, max_queue_size=10)


testGen.close()


from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_fscore_support as score

y_pred = np.array(predictions)
y_valid = np.array(testGen.db["labels"])
matrix = confusion_matrix(y_valid, y_pred.argmax(axis=1), labels = [0,1,2,3,4])

f1 = f1_score(y_valid, y_pred.argmax(axis=1), average = 'micro')
print("micro f1:")
print("f1_score", f1_score(y_valid, y_pred.argmax(axis=1), average=None))
print("precision_score", precision_score(y_valid, y_pred.argmax(axis=1), average=None))

print("recall_score", recall_score(y_valid, y_pred.argmax(axis=1), average=None))  


precision, recall, fscore, support = score(y_valid, y_pred.argmax(axis=1))
print("\n\nmacro f1:")
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print('precision: {}'.format(np.mean(precision)))
print('recall: {}'.format(np.mean(recall)))
print('fscore: {}'.format(np.mean(fscore)))
print('support: {}'.format(np.mean(support)))





