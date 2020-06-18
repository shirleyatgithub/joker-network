# joker-network
connect hardware

open two terminal

run consumerudp.py 
  python consumerudp.py cnnModelName resize arduinoPort
  example: python consumerudp.py afnorm224v1.h5 224 arduinoPort

run producerudp.py
  python producerudp.py resize
  example: python producerudp.py 224

open ArduinoControl/timerpwm2/timerpwm2.ino
