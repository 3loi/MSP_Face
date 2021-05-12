#!/usr/bin/env python
# coding: utf-8


# import the necessary packages
from config_file import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.io import HDF5DatasetWriter
import numpy as np
import progressbar
import json
import cv2
import os
import pandas as pd


#AffectNet Classes
# 0 - Neutral
# 1 - Happy
# 2 - Sad
# 3 - Surprised
# 4 - Fear
# 5 - Disgust
# 6 - Anger
# 7 - Contempt
# 8 - None
# 9 - Uncertain
# 10- Non-Face


if NUM_CLASSES is 5:
    classes = [0, 1, 2, 5, 6]
else:
    classes = [0, 1, 2, 3, 4, 5, 6, 7]




# grab the paths to the images
df_valid = pd.read_csv(INPUT_PATH_VALID)
df_train = pd.read_csv(INPUT_PATH_TRAIN)

df_valid = df_valid.rename(index=str, columns={"subDirectory_filePath": "path"})
df_train = df_train.rename(index=str, columns={"subDirectory_filePath": "path"})

df_valid = df_valid[df_valid.expression.isin(classes)]
df_train = df_train[df_train.expression.isin(classes)]

trainPaths = list(df_train.path)
trainLabels = list(df_train.expression)

testPaths = list(df_valid.path)
testLabels = list(df_valid.expression)

#rename labels to make sure the label order is sequential (i.e., 1, 2, 3, 4)
if NUM_CLASSES is 5:
    df_train['expression'][df_train['expression'] == 6] = 3
    df_valid['expression'][df_valid['expression'] == 6] = 3

    df_train['expression'][df_train['expression'] == 5] = 4
    df_valid['expression'][df_valid['expression'] == 5] = 4

#takes a dataframe returns paths/labels, each lable having a maximum 'size' samples for each label/category
def downsample(df, size):
    counts = [0 for i in classes]
    l = []
    p = []
    for num, row in df.iterrows():
        if counts[row.expression] < size:
          counts[row.expression] += 1
          l.append(row.expression)
          p.append(row.path)
    return l, p

#downsample to a fixed number
testLabels, testPaths = downsample(df_valid, 1500000)
trainLabels, trainPaths = downsample(df_train, 24882)

testPaths, testLabels = shuffle(testPaths, testLabels)
trainPaths, trainLabels = shuffle(trainPaths, trainLabels)


print("Number of Images in each category")
for i in range(config.NUM_CLASSES):
    print(i, trainLabels.count(i))

split = train_test_split(trainPaths, trainLabels,
	test_size=NUM_VAL_IMAGES, stratify=trainLabels,
	random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split



datasets = [
	("train", trainPaths, trainLabels, TRAIN_HDF5),
	("val", valPaths, valLabels, VAL_HDF5),
	("test", testPaths, testLabels, TEST_HDF5)]




aap = AspectAwarePreprocessor(IMAGE_HEIGHT, IMAGE_WIDTH)
(R, G, B) = ([], [], [])



# loop over the dataset tuples
IMAGE_PATH = IMAGES_PATH
for (dType, paths, labels, outputPath) in datasets:
	# create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), IMAGE_HEIGHT, IMAGE_HEIGHT, 3), outputPath)

	# initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
		progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths),
		widgets=widgets).start()

	# loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        try:
            image = cv2.imread( os.path.join(IMAGE_PATH, path))
            image = aap.preprocess(image)
        except:
            print("Unable to load Image: {}".format(os.path.join(IMAGE_PATH, path)))
            continue

		# if we are building the training dataset, then compute the
		# mean of each channel in the image, then update the
		# respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

		# add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)

	# close the HDF5 writer
    pbar.finish()
    writer.close()



# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()






