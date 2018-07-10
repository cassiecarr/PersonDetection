import numpy as np
from keras.layers import Input

# define parameters 
class my_parameters:
	LABELS = ['person']
	IMG_H, IMG_W = 416, 416
	GRID_H, GRID_W = 13 , 13
	BOX = 5
	CLASS = len(LABELS)
	CLASS_WEIGHTS = np.ones(CLASS, dtype='float32')
	OBJ_THRESHOLD    = 0.3
	NMS_THRESHOLD    = 0.3
	ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

	NO_OBJECT_SCALE  = 1.0
	OBJECT_SCALE     = 5.0
	COORD_SCALE      = 1.0
	CLASS_SCALE      = 1.0

	BATCH_SIZE       = 16
	WARM_UP_BATCHES  = 0
	TRUE_BOX_BUFFER  = 50

	input_image = Input(shape=(IMG_H, IMG_W, 3))
	true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))