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
from preprocessing import parse_annotation, BatchGenerator
from utils import WeightReader, decode_netout, draw_boxes, compute_overlap, compute_ap
from keras.models import load_model
from custom_loss import custom_loss
from config import my_parameters

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def evaluate(model, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet
        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        # gather all detections and annotations
        all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
        all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

        for i in range(generator.size()):
            raw_image = generator.load_image(i)
            raw_height, raw_width, raw_channels = raw_image.shape

            dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
            raw_image = cv2.resize(raw_image, (416, 416))
            raw_image = raw_image / 255.
            raw_image = raw_image[:,:,::-1]
            raw_image = np.expand_dims(raw_image, 0)
            
            # make the boxes and the labels
            pred_boxes  = model.predict([raw_image, dummy_array])
            boxes = decode_netout(pred_boxes[0], 
                      obj_threshold=OBJ_THRESHOLD,
                      nms_threshold=NMS_THRESHOLD,
                      anchors=ANCHORS, 
                      nb_class=CLASS)

            
            score = np.array([box.get_score() for box in boxes])
            pred_labels = np.array([box.get_label() for box in boxes])        
            
            if len(boxes) > 0:
                boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.get_score()] for box in boxes])
            else:
                boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            boxes  = boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(generator.num_classes()):
                all_detections[i][label] = boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions   

############################################
# Compute mAP on the validation set
############################################

# define generator config
generator_config = {
    'IMAGE_H'         : IMG_H, 
    'IMAGE_W'         : IMG_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'BOX'             : BOX,
    'LABELS'          : LABELS,
    'CLASS'           : len(LABELS),
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}
# define paths for training                     
valid_image_folder = 'data/Validation_Images/'
valid_annot_folder = 'data/Validation_Annotations/'

# define normalize for images
def normalize(image):
    return image / 255.

# define the image and label datasets for training
valid_imgs, valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=LABELS)
valid_generator = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)

# load new model from training
model = load_model('new_model_6.h5', custom_objects={
        'tf':tf,
        'custom_loss':custom_loss
})

average_precisions = evaluate(model, valid_generator)     

 # print evaluation
print('mAP: {:.4f}'.format(sum(average_precisions.values()) / len(average_precisions)))  
