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
preTrainedModel = sys.argv[2]
resize = int(sys.argv[3])



classnum=2


model = load_model(preTrainedModel)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


train_batch_size = 128
valid_batch_size = 64
test_batch_size = 32

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

test_datagen = ImageDataGenerator(rescale=1./255,) #测试集不做增强
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



import datetime
print("{} Start testing...".format(datetime.datetime.now()))

train_generator.reset()
loss, acc = model.evaluate_generator(train_generator, steps=100, verbose=2)

print("pretrained model, train accuracy: {:5.2f}%".format(100*acc))
print("{} end testing".format(datetime.datetime.now()))

valid_generator.reset()
loss, acc = model.evaluate_generator(valid_generator, verbose=2)
print("pretrained model, valid accuracy: {:5.2f}%".format(100*acc))
print("{} end testing".format(datetime.datetime.now()))
