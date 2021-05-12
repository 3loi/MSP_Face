from os import path

# define the paths to the images directory (AffectNet Corpus)
IMAGES_PATH = "/storage/cropped_Annotated"

# used when creating the dataset, user None for VAD 
NUM_CLASSES = 8
# NUM_CLASSES = 5
# NUM_CLASSES = None

#number of images to set aside for validation
NUM_VAL_IMAGES = 200*NUM_CLASSES

# use the base path to define the path to the input emotions file
INPUT_PATH_VALID = path.sep.join([IMAGES_PATH, "validation.csv"])
INPUT_PATH_TRAIN = path.sep.join([IMAGES_PATH, "training.csv"])


# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "./train8emo.hdf5"
VAL_HDF5 = "./val8emo.hdf5"
TEST_HDF5 = "./test8emo.hdf5"

BATCH_SIZE = 32
EPOCHS = 60


IMAGE_WIDTH = 128 
IMAGE_HEIGHT = 128
# IMAGE_WIDTH = 224 
# IMAGE_HEIGHT = 224

# path to the output model file
MODEL_PATH = "output/affectnet_8class.model"

# define the path to the dataset mean
DATASET_MEAN = "output/affectnet_emo_mean_8class.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "/home/user/Desktop/affectnet/"


