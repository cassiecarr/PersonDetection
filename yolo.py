from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.models import load_model

# path to the model weights files.
weights_path = 'model-data/yolo.h5'
top_model_weights_path = 'model-data/my_yolo.h5'

# define   
ann_dir = 'data/ILSVRC/Annotations/DET/train/ILSVRC2014_train_0000/'
img_dir = 'data/ILSVRC/Data/DET/train/ILSVRC2014_train_0000/'

NORM_H, NORM_W = 416, 416
GRID_H, GRID_W = 13 , 13
BATCH_SIZE = 8
BOX = 5
ORIG_CLASS = 20


# # dimensions of our images.
# img_width, img_height = 150, 150

# train_data_dir = 'cats_and_dogs_small/train'
# validation_data_dir = 'cats_and_dogs_small/validation'
# nb_train_samples = 2000
# nb_validation_samples = 800
# epochs = 50
# batch_size = 16

# import the YOLO model
yolo_model = load_model(weights_path)

# remove the last layer from the imported YOLO model
yolo_model.layers.pop()
num_frozen_layers = len(yolo_model.layers)

# print the summery of the model
yolo_model.summary()

# build a classifier model to put on top of ybhgthe convolutional model
top_model = Sequential()
top_model.add(Conv2D(BOX * (4 + 1 + ORIG_CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
top_model.add(Activation('linear'))
top_model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + ORIG_CLASS)))

print(yolo_model.output_shape[1])
# input_shape=yolo_model.output_shape[1:]

# new_yolo_model = Model(inputs=yolo_model.input, outputs=top_model(yolo_model.output))

# # set the original layers to ,non-trainable (weights will not be updated)
# for layer in model.layers[:num_frozen_layers]:
#     layer.trainable = False

# print(num_frozen_layers)
# print(len(new_yolo_model.layers))

# # compile the model with a SGD/momentum optimizer
# # and a very slow learning rate.
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

# # prepare data augmentation configuration
# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='binary')

# # fine-tune the model
# model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     epochs=epochs,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)