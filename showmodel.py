"""DAVIS346 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
这个版本是把predict作为主进程，但是producer在产生了4次数据后，就get不到数据了。
Device ID: 1
Device is master.
Device Serial Number: 00000002
Device String: DAVIS ID-1 SN-00000002 [4:12]
Device USB bus Number: 4
Device USB device address: 12
Device size X: 346
Device size Y: 260
Logic Version: 18
Background Activity Filter: True
Color Filter 0 <class 'int'>
False
frame shape 0
frame shape 0
get_event
data not none
produce time 11.95 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 68.867 ms
write ard time 0.107 ms
predition done
produce time 11.367 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.377 ms
write ard time 0.041 ms
predition done
produce time 7.382 ms
classify recv msg (224, 224, 1)
get_event
data not none
index 0 predict time 4.385 ms
write ard time 0.039 ms
predition done
produce time 5.541 ms
classify recv msg (224, 224, 1)
index 0 predict time 4.105 ms
write ard time 0.046 ms
predition done
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
get_event
"""
from __future__ import print_function

import cv2
import numpy as np

# from pyaer import libcaer
# from pyaer.davis import DAVIS

from datetime import datetime


import cv2
import sys,os
import gc
import tensorflow as tf
# from face_train_use_keras import Model
# from model import create_smodel, create_model
import time

import keras
from keras.models import load_model


#!/usr/bin/python
import serial
import syslog
import time
from datetime import datetime
import math

# from functools import reduce

import numpy.ma as ma

# import multiprocessing,os,time
# from multiprocessing import Process


# import socket
# from utils import recv_into
# import numpy as np
# MAXB = 10000000


# PORT = 12000
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# address = ("", PORT)
# server_socket.bind(address)


#The following line is for serial over GPIO


# savepath = './dvsoutrec2/tmp0/'

class_name = ['joker', 'other']

#人脸识别分类器本地存储路径
cascade_path = "./jokeravi.xml"   

# modelpath = './184binv32.h5'
# modelpath = './alexnet184grayv12.h5'
modelpath = sys.argv[1]

     
color = (0, 255, 0)

model = load_model(modelpath)
model.summary()