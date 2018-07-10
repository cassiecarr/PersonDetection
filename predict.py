# REF: This was adapted from https://github.com/experiencor/keras-yolo2

# intialization
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
from keras import initializers
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import imgaug as ia
from tqdm import tqdm
from imgaug import augmenters as iaa
import numpy as np
import pickle
import os, cv2
from utils import WeightReader, decode_netout, draw_boxes
from keras.models import load_model
from custom_loss import custom_loss
from config import my_parameters

# define parameters from config.py
LABELS = my_parameters.LABELS
IMG_H, IMG_W = my_parameters.IMG_H, my_parameters.IMG_W
GRID_H, GRID_W = my_parameters.GRID_H, my_parameters.GRID_W
BOX = my_parameters.BOX
CLASS = my_parameters.CLASS
CLASS_WEIGHTS = my_parameters.CLASS_WEIGHTS
OBJ_THRESHOLD    = my_parameters.OBJ_THRESHOLD
NMS_THRESHOLD    = my_parameters.NMS_THRESHOLD
ANCHORS          = my_parameters.ANCHORS

NO_OBJECT_SCALE  = my_parameters.NO_OBJECT_SCALE
OBJECT_SCALE     = my_parameters.OBJECT_SCALE
COORD_SCALE      = my_parameters.COORD_SCALE
CLASS_SCALE      = my_parameters.CLASS_SCALE

BATCH_SIZE       = my_parameters.BATCH_SIZE
WARM_UP_BATCHES  = my_parameters.WARM_UP_BATCHES
TRUE_BOX_BUFFER  = my_parameters.TRUE_BOX_BUFFER

input_image = my_parameters.input_image
true_boxes  = my_parameters.true_boxes

# load new model from training
model = load_model('new_model_3.h5', custom_objects={
    	'tf':tf,
    	'custom_loss':custom_loss
})

# load image
image = cv2.imread('data/test_img/000229.jpg')
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

input_image = cv2.resize(image, (416, 416))
input_image = input_image / 255.
input_image = input_image[:,:,::-1]
input_image = np.expand_dims(input_image, 0)

netout = model.predict([input_image, dummy_array])

boxes = decode_netout(netout[0], 
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS, 
                      nb_class=CLASS)
            
image = draw_boxes(image, boxes, labels=LABELS)

cv2.imwrite('data/test_predicted_img/000229.jpg',image)

