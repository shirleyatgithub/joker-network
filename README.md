# joker-network
connect hardware [optional]
open two terminal

how to run consumerudp.py
python consumerudp.py cnnModelName resize [arduinoPort]
example: python consumerudp.py afnorm224v1.h5 224 [arduinoPort]
arduinoPort optional, if no arduino board connected, no port.

run producerudp.py
python producerudp.py resize
example: python producerudp.py 224

open ArduinoControl/timerpwm2/timerpwm2.ino

joker-classify-network.py is for training.  
command python joker-classify-network.py imgdir weightfilename resize
imgdir is where your imgs are organized with three sub-folders: train/, valid/, and test/.  
Each sub-folder contains two sub-folders, class1/ and class2/. Class1/ includes non-joker images and class2/ contains joker images.

joker-classify-accuracy.py is for getting accuracy with pre-trained model.  
