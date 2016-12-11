#!/usr/bin/env python

from __future__ import division
from squeezenet import SqueezeNet
from keras.utils.np_utils import to_categorical
import numpy as np
import os
import cv2
import random

images_dir = './images'
weights_file = './weights.h5'
initial_epoch = 0
nb_epoch = 1
batch_size = 64
validation_split = 0.2 

class_mapping = {
    'coca_cola_bottles': 0,
    'fanta_bottle': 1,
    'cola_cans': 2, 
    'fanta_cans': 3,
    'paper_coffee_cups': 4,
    'water_bottles': 5
}
nb_classes = len(class_mapping)


def load_image(img_path):
    # Load image with 3 channel colors
    img = cv2.imread(img_path, flags=1)
    # img = img_path
    # print img.shape

    # Crop image to a square
    height = img.shape[0]
    width = img.shape[1]
    offset = int(round(max(height, width) / 2.0))

    padded_img = cv2.copyMakeBorder(img, 227, 227, 227, 227, cv2.BORDER_CONSTANT)
    padded_height = padded_img.shape[0]
    padded_width = padded_img.shape[1]
    center_x = int(round(padded_width / 2.0))
    center_y = int(round(padded_height / 2.0))
    
    cropped_img = padded_img[center_y - offset: center_y + offset, center_x - offset: center_x + offset]

    # Resize image to 227, 227 as Squeezenet only accepts this format.
    resized_image = cv2.resize(cropped_img, (227, 227)).astype(np.float32)
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image

# List comprehension returns list of tuples (image_path, classification)
imgpaths_classes = [ (os.path.join(subdir, f), os.path.basename(subdir)) 
                        for subdir, dirs, files in os.walk(images_dir) 
                            for f in files ]
# Randomize it. 
random.shuffle(imgpaths_classes)
# Split into training and validation set. 
split_index = int(validation_split * len(imgpaths_classes))
validation_images = imgpaths_classes[:split_index]
training_images  = imgpaths_classes[split_index:]

samples_per_epoch = len(training_images) - 10
nb_val_samples = len(validation_images) - 10


# Generator expression. Yields two tuples (image, class). Use generator because images might not fit into memory,
training_data = ( (load_image(img_path), to_categorical([class_mapping[classification]], nb_classes=nb_classes)) 
                    for img_path, classification in training_images )

validation_data = ( (load_image(img_path), to_categorical([class_mapping[classification]], nb_classes=nb_classes)) 
                        for img_path, classification in validation_images )

# Unzip to two lists.
# images, classes = zip(*images_classes)
# [images, classes], [x, y] = mnist.load_data()
# images = images[0:500]
# classes = classes[0:500]

# images = np.array([cv2.resize(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB), (227, 227)) for im in images])
# images = np.array(images)
# print images.shape
# classes = to_categorical(classes, nb_classes=nr_classes)

print('Loading model..')
model = SqueezeNet(nb_classes)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
if os.path.isfile(weights_file):
    print('Loading weights: %s' % weights_file)
    model.load_weights(weights_file, by_name=True)

print('Fitting model')
# model.fit(images, classes, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.2, initial_epoch=0)
model.fit_generator(training_data, 
        samples_per_epoch=samples_per_epoch, 
        validation_data=validation_data, 
        nb_val_samples=nb_val_samples, 
        nb_epoch=nb_epoch, 
        verbose=1, 
        initial_epoch=initial_epoch)
print("Finished fitting model")

print('Saving weights')
model.save_weights(weights_file, overwrite=True)
print('Evaluating model')
# score = model.evaluate(images, classes, verbose=1)

validation_data = ( (load_image(img_path), to_categorical([class_mapping[classification]], nb_classes=nb_classes)) 
                        for img_path, classification in validation_images )
score = model.evaluate_generator(validation_data, val_samples=nb_val_samples)
print('result: %s' % score)

