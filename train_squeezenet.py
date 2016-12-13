#!/usr/bin/env python

from __future__ import division
from squeezenet import SqueezeNet
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import random

images_dir = './images'
weights_file = './weights.h5'
initial_epoch = 0
nb_epoch = 1
batch_size = 64
samples_per_epoch=1300
nb_val_samples=100

class_mapping = {
    'coca_cola_bottles': 0,
    'fanta_bottle': 0,
    'cola_cans': 1, 
    'fanta_cans': 1,
    'paper_coffee_cups': 2,
    'water_bottles': 0
}

classes = [ 'coca_cola_bottles', 'fanta_bottle', 'cola_cans', 'fanta_cans', 'paper_coffee_cups', 'water_bottles']

nb_classes = len(classes)

train_datagen = ImageDataGenerator(
        rescale=1./255, 
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory('./images', 
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical', 
        classes=classes
        )

val_generator = test_datagen.flow_from_directory('./images', 
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical', 
        classes=classes
        )



print('Loading model..')
model = SqueezeNet(nb_classes)
# adam = Adam(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
if os.path.isfile(weights_file):
    print('Loading weights: %s' % weights_file)
    model.load_weights(weights_file, by_name=True)

print('Fitting model')
model.fit_generator(train_generator, 
        samples_per_epoch=samples_per_epoch, 
        validation_data=val_generator, 
        nb_val_samples=nb_val_samples, 
        nb_epoch=nb_epoch, 
        verbose=1, 
        initial_epoch=initial_epoch)
print("Finished fitting model")

print('Saving weights')
model.save_weights(weights_file, overwrite=True)
print('Evaluating model')
score = model.evaluate_generator(val_generator, val_samples=nb_val_samples)
print('result: %s' % score)

