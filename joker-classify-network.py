from __future__ import absolute_import, division, print_function, unicode_literals

import os,sys
import shutil, random, glob
import cv2
import numpy as np
import pandas as pd

import math

import matplotlib.pyplot as plt
from glob import glob

import tensorflow as tf
from tensorflow import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras.models import Model, load_model
from keras.utils import np_utils


print(tf.version.VERSION)

imgdir = sys.argv[1] 
print('dataset path', imgdir)
weightModelName = sys.argv[2]
resize = int(sys.argv[3])

# classification task categories: 2, joker and non-joker.
# network output: 2 neurons. (1,0) means non-joker, (0,1) means joker
classnum=2

# model definition

#LeNet-5
# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(5,5), 
#                  padding='valid', 
#                  input_shape=(184, 184, 3), 
#                  activation='tanh'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=16, kernel_size=(5,5), 
#                  padding='valid', 
#                  activation='tanh')) #
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(120, activation='tanh'))
# model.add(Dense(84, activation='tanh'))
# model.add(Dense(2, activation='softmax')) #softmax函数做激活函数 e^x/(sum(e^x))

# sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=True) #采用随机梯度下降法作为优化算法
# model.compile(loss='binary_crossentropy',
#               optimizer=sgd, 
#               metrics=['accuracy'])

# AlexNet modified
def create_model():
    model = Sequential()

    # model.add(Input(shape=(None,resize,resize,3),dtype='float32', name='input'))
    model.add(Conv2D(filters=64, kernel_size=(11,11),
                     strides=(4,4), padding='valid',
                     input_shape=(resize,resize,1),
                     activation='relu', name='conv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), 
                           padding='valid'))

    model.add(Conv2D(filters=64, kernel_size=(5,5), 
                     strides=(1,1), padding='same', 
                     activation='relu', name='conv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), 
                           padding='valid'))

    model.add(Conv2D(filters=128, kernel_size=(3,3), 
                     strides=(1,1), padding='same', 
                     activation='relu', name='conv3'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), 
                     strides=(1,1), padding='same', 
                     activation='relu', name='conv4'))
    #model.add(Conv2D(filters=256, kernel_size=(3,3), 
    #                 strides=(1,1), padding='same', 
    #                 activation='relu', name='conv5'))

    model.add(MaxPooling2D(pool_size=(3,3), 
                           strides=(2,2), padding='valid'))

    model.add(Flatten())
    # model.add(Dense(4096, activation='relu', name='fc6'))
    # model.add(Dropout(0.5))

    # model.add(Dense(4096, activation='relu', name='fc7'))
    # model.add(Dropout(0.5))

    model.add(Dense(100, activation='relu', name='fc8'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(classnum, activation='softmax', name='output'))

    return model

model = create_model()

# print the model parameter
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# batch size
train_batch_size = 128
valid_batch_size = 64
test_batch_size = 32

# data generator for reading images from image folder and data-augmentation

train_datagen = ImageDataGenerator( #实例化
    rescale=1./255,
    rotation_range = 90,  #图片随机转动的角度
    width_shift_range = 0.2, #图片水平偏移的幅度
    height_shift_range = 0.2, #图片竖直偏移的幅度
    zoom_range = 0.2,
    horizontal_flip=True) #随机放大或缩小


train_generator = train_datagen.flow_from_directory(
        imgdir + '/train/',
        target_size=(resize, resize),
        batch_size=train_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1./255,) # no data augmentation for valid and test set

valid_generator = test_datagen.flow_from_directory(
        imgdir + '/valid/',
        target_size=(resize, resize),
        batch_size=valid_batch_size,
        class_mode='categorical',
        color_mode='grayscale')

test_generator = test_datagen.flow_from_directory(
        imgdir + '/test/',
        target_size=(resize, resize),
        batch_size=test_batch_size,
        class_mode='categorical',
        color_mode='grayscale')


# train settings
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# train process
result = model.fit_generator(train_generator, 
          steps_per_epoch=300, 
          epochs=100, verbose=1,
          validation_data=valid_generator,
          validation_steps=25,
          callbacks=[mcp_save],
          # max_queue_size=capacity,
          shuffle = True,
          workers=1)

# save model
model.save(weightModelName)