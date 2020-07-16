"""trixie producer: getting data from davis346, generating frames based on 3-sigma rule,
and sending the data to consumer.

Author: Shasha Guo
Email : shirleyguo19@gmail.com
"""
from __future__ import print_function

import cv2
import numpy as np

from pyaer import libcaer
from pyaer.davis import DAVIS

from datetime import datetime


import cv2
import sys,os
import gc
# import tensorflow as tf
# from face_train_use_keras import Model
# from model import create_smodel, create_model
import time



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
import numpy as np
from utils import send_from
#SOCK_DGRAM udp mode

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('', 12000)

class_name = ['joker', 'other']
# the size of frame to be resized to
resize = int(sys.argv[1])

count = 0

# savedcounter = 30000

device = DAVIS(noise_filter=True)

print("Device ID:", device.device_id)
if device.device_is_master:
    print("Device is master.")
else:
    print("Device is slave.")
print("Device Serial Number:", device.device_serial_number)
print("Device String:", device.device_string)
print("Device USB bus Number:", device.device_usb_bus_number)
print("Device USB device address:", device.device_usb_device_address)
print("Device size X:", device.dvs_size_X)
print("Device size Y:", device.dvs_size_Y)
print("Logic Version:", device.logic_version)
print("Background Activity Filter:",
      device.dvs_has_background_activity_filter)
print("Color Filter", device.aps_color_filter, type(device.aps_color_filter))
print(device.aps_color_filter == 1)

device.start_data_stream()
# setting bias after data stream started
device.set_bias_from_json("./configs/davis346_config.json")

clip_value = 3
histrange = [(0, v) for v in (260, 346)]

def get_event(device):
    data = device.get_event()
    return data

num_packet_before_disable = 0

rectifyPolarities = True

rangeNormalizeFrame = 255

# constant event number to form a frame
CONSTNUM = 3000

frame = np.zeros([])
print('frame shape', len(frame.shape))

img2 = np.zeros([])

def producer():

    frame = np.zeros([])
    print('frame shape', len(frame.shape))
    while True:
        try:
            i = 0

            # print('---------------------------------fetch data')
            leng = len(frame.shape)
            a = datetime.now()
            data = get_event(device)
            print('get_event')
            if data is not None:
                print('data not none')
                (pol_events, num_pol_event,
                 special_events, num_special_event,
                 frames_ts, frames, imu_events,
                 num_imu_event) = data

                if num_pol_event != 0:

                    # get enough events to form a frame as parameter indicated

                    if leng > 0:
                        # if the packet is larger than the number, discard the extra events
                        if frame.shape[0] > CONSTNUM:
                            frame = frame[:CONSTNUM]
                            # print('--------------------enough data')
                        else:
                            # if the packet is smaller than the number, store the events and wait for next packet
                            frame = np.vstack([frame, pol_events])

                            if frame.shape[0] > CONSTNUM:
                          
                                frame = frame[:CONSTNUM]
                                
                            else:
                                continue
                    else:

                        frame = pol_events

                        if frame.shape[0] > CONSTNUM:
                              
                            frame = frame[:CONSTNUM]
                           
                        else:
                           
                            continue

                    pol_on = (frame[:, 3] == 1)
                    pol_off = np.logical_not(pol_on)

                    img_on, _, _ = np.histogram2d(
                            frame[pol_on, 2], frame[pol_on, 1],
                            bins=(260, 346), range=histrange)
                    img_off, _, _ = np.histogram2d(
                            frame[pol_off, 2], frame[pol_off, 1],
                            bins=(260, 346), range=histrange)

        
                    imgtmp = img_on - img_off
                    tmpvar = np.reshape(imgtmp, [imgtmp.shape[0] * imgtmp.shape[1],1]).astype('uint8')
                    pixmap = np.zeros(tmpvar.shape)

                    n = len(pixmap)

                    tmpvar = tmpvar * 1. / np.max(tmpvar)
                    tmpvar2 = ma.masked_values(tmpvar, 0)
                    var3 = np.var(tmpvar2.compressed())

                    sig = math.sqrt(var3)
                    if sig < (0.1 / 255.0):
                        sig = 0.1 / 255.0

                    pixmap = np.zeros(tmpvar.shape)

                    # print('pixmap', pixmap.shape)

                    numSDevs = 3.
                    mean_png_gray = 0 if rectifyPolarities == True else (127. / 255)
                    
                    zeroValue = mean_png_gray
                    fullscale = 1. - zeroValue
                    
                    fullrange = numSDevs * sig if rectifyPolarities == True else (2. * numSDevs * sig)
                    halfRange = 0. if rectifyPolarities == True else (numSDevs * sig)
                    rangenew = 1.

                    n = len(pixmap)

                    tmpvar[tmpvar > 0.] = ((tmpvar[tmpvar > 0.] + halfRange)*rangenew) / fullrange
                    tmpvar[tmpvar > 1.] = 1.
                    tmpvar[tmpvar == 0.] = mean_png_gray

                    pixmap = tmpvar * 1.0 * rangeNormalizeFrame

                    # print('tmpvar unique after', np.unique(pixmap))

                    fimg = np.reshape(pixmap, imgtmp.shape)

                    img1 = fimg.astype('uint8')

                    b = datetime.now()

                    img4 = cv2.resize(img1, (resize, resize), interpolation=cv2.INTER_NEAREST)

                    print('process time', (b-a).microseconds / 1000, 'ms')

                    try:
                        # sending data to consumer
                        a = datetime.now()
                        data = img4.tostring()
                        client_socket.sendto(data, server_address)
                        b = datetime.now()
                        print('communication time', (b-a).microseconds / 1000, 'ms')

                        print(a)# decode()
                    except KeyboardInterrupt:
                        print('interrupted!')
                        break


                    # conn_a.send(img4)
                    

                    frame = np.zeros([])


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                pass

        except KeyboardInterrupt:
            device.shutdown()
            break


if __name__ == '__main__':


    try:
        producer()
    except Exception as e:
        print('Error', str(e))
        device.shutdown()
        sys.exit()
    else:
        pass
    finally:
        pass