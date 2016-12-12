#!/usr/bin/env python

from __future__ import division
from squeezenet import SqueezeNet
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import os
import cv2
import random
import uuid

images_dir = './images'
weights_file = './mnist_weights.h5'
initial_epoch = 0
nb_epoch = 2
batch_size = 64
validation_split = 0.2 
input_shape = (67, 67, 3)

nb_classes = 10


def load_image(img):
    # Load image with 3 channel colors
    # img = cv2.imread(img_path, flags=1)
    name = str(uuid.uuid4())
    # Convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Image needs to the resized to (227x227), but we want to maintain the aspect ratio.
    # height = img.shape[0]
    # width = img.shape[1]
    # offset = int(round(max(height, width) / 2.0))

    # # Add borders to the images.
    # padded_img = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT)
    # padded_height = padded_img.shape[0]
    # padded_width = padded_img.shape[1]
    # center_x = int(round(padded_width / 2.0))
    # center_y = int(round(padded_height / 2.0))
    # # Crop the square containing the full image.
    # cropped_img = padded_img[center_y - offset: center_y + offset, center_x - offset: center_x + offset]

    # Resize image to 227, 227 as Squeezenet only accepts this format.
    resized_image = cv2.resize(img, (input_shape[0], input_shape[1])).astype(np.float64)
    return resized_image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Shuffle lists
train_zipped = zip(X_train, y_train)
test_zipped = zip(X_test, y_test)
random.shuffle(train_zipped)
random.shuffle(test_zipped)

X_train[:], y_train[:] = zip(*train_zipped)
X_test[:], y_test[:] = zip(*test_zipped)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

def gen(x, y): 
    nb = 100
    for i in xrange(len(x)):
        # yield (np.expand_dims(x[i], axis=0), np.expand_dims(y[i], axis=0))
        yield (np.asarray([load_image(img) for img in x[i*nb:i*nb+nb]]), y[i*nb:i*nb+nb])

# the data, shuffled and split between train and test sets
# train, test = mnist.load_data()
# train = zip(*train)
# test = zip(*test)

samples_per_epoch = 1000 #len(training_images) - 20
nb_val_samples = 200 # len(validation_images) - 20

# Generator expression. Yields two tuples (image, class). Use generator because images might not fit into memory,
# training_data = ( (load_image(x), to_categorical([y], nb_classes=nb_classes)) for x, y in train )
# validation_data = ( (load_image(x), to_categorical([y], nb_classes=nb_classes)) for x, y in test )
# training_data = gen(X_train, Y_train)
# validation_data = gen(X_test, Y_test)

training_data = gen(X_train, Y_train)
validation_data = gen(X_test, Y_test)

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
model = SqueezeNet(nb_classes, input_shape=input_shape)
adam = Adam(lr=0.0040)
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
if os.path.isfile(weights_file):
    print('Loading weights: %s' % weights_file)
    model.load_weights(weights_file, by_name=True)

total = 0
correct = 0

for images, classes in training_data: 
    for i in xrange(len(images)):
        img = np.expand_dims(images[i], axis=0)
        cl = classes[i]
        cl = cl.argsort()[-1:][::-1]
        result = model.predict(img)
        res = result[0].argsort()[-1:][::-1]
        if res[0] == cl[0]:
            correct += 1
        total += 1
        print('result: %s / %s, correct: %s/%s' % (res, cl, correct, total))

