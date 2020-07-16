"""DAVIS346 test example.

Author: Shasha Guo
Email : shirleyguo19@gmail.com
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

import multiprocessing,os,time
from multiprocessing import Process


import socket
from utils import recv_into
import numpy as np
MAXB = 10000000

# port of socket
PORT = 12000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
address = ("", PORT)
server_socket.bind(address)


#The following line is for serial over GPIO


class_name = ['joker', 'other']

# the with file with postfix .h5, e.g. afnorm224v1.h5
modelpath = sys.argv[1]
# the size of frame to be resized to
resize = int(sys.argv[2])
# the arduino port
ardFlag = 0
if len(sys.argv) > 3:
    port = sys.argv[3]
    ard = serial.Serial(port, 9600, timeout=5)
    ardFlag = 1

model = load_model(modelpath)

count = 0

# savedcounter = 30000


clip_value = 3
histrange = [(0, v) for v in (260, 346)]


shapelike = [260,346,1]
def get_event(device):
    data = device.get_event()

    return data


num_packet_before_disable = 0


frame = np.zeros([])
print('frame shape', len(frame.shape))

img2 = np.zeros([])

pixelnum = resize * resize
print(pixelnum)

if __name__ == '__main__':
    while True:

        try:
            
            stime = datetime.now()

            receive_data, client_address = server_socket.recvfrom(pixelnum)
            # print(len(receive_data), type(receive_data))
            etime = datetime.now()
            # print('load image', (etime - stime).microseconds / 1000, 'ms')
            # receive_data = receive_data.decode("cp1252")
            # print(len(receive_data))
            stime = datetime.now()
            receive_data = np.fromstring(receive_data, dtype=np.uint8)
            # print(receive_data)
            # print("client %s sending data: %s" % (client_address, np.reshape(receive_data,[3,3,1])))
            img1 = np.reshape(receive_data, [resize, resize])
            # print('recv from udp', img1.shape)
            # img1 = cv2.imread(savepath + 'newest.jpg', 0)

            img4 = np.reshape(img1, (resize, resize, 1))

            img4 = img4 * 1. / 255
            

            etime = datetime.now()


            stime = datetime.now()

            tmp = model.predict(img4[None, :]) 

            etime = datetime.now()

            pred = list(tmp[0])
                
            index = pred.index(max(pred))
                
            # print('index', index, 'predict time', (etime - stime).microseconds / 1000, 'ms')
            # label = class_name[pred.index(max(pred))]

            if ardFlag:
                stime = datetime.now()
                setTemp1 = str(index)
                ard.write(setTemp1.encode())
                etime = datetime.now()
            
            # print('write ard time', (etime - stime).microseconds / 1000, 'ms')

            # time.sleep(0.005)

            cv2.imshow("image", img4)
            if index != 0:
                print('finger index', index)
                # if joker, stop 0.5ms for show
                time.sleep(0.5)

                # write the frames into jpg
                # stime = datetime.now()
                # jpgname = savepath + str(savedcounter) + '.jpg'
                # cv2.imwrite(jpgname, img1)

                # savedcounter += 1

                # etime = datetime.now()
                # print('storage time', (etime - stime).microseconds / 1000, 'ms')

            # print('prediction done')

        except Exception as e:
            print(str(e))
