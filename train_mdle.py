import numpy as np
import pyautogui
import cv2

from PIL import ImageGrab
from time import sleep

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3

def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model




def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True,
              run_id='openai_learning')
    
    return model

data = []
cl = [1,0]
lv = [0,1]
click = pyautogui.rightClick()

while(True):
#    screen = np.array( ImageGrab.grab(bbox=(0,40,800,500)))
#
#    cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
#    processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)


    magnification = 5
    mx, my = pyautogui.position()  # get the mouse cursor position
    x = mx - 15  # move to the left 15 pixels
    y = my - 15  # move up 15 pixels
    capture = ImageGrab.grab(
                  bbox=(x, y, x + 30, y + 30)
              )  # get the box down and to the right 15 pixels (from the cursor - 30 from the x, y position)
    arr = np.array(capture)  # convert the image to numpy array
    res = cv2.cvtColor(
              cv2.resize(
                  arr, 
                  None, 
                  fx=magnification, 
                  fy=magnification, 
                  interpolation=cv2.INTER_CUBIC
              ), cv2.COLOR_BGR2GRAY
          )  # magnify the screenshot and convert to grayscale

    cv2.imshow('window_mag' , cv2.cvtColor(res, cv2.COLOR_BGR2RGB))



    if cv2.waitKey(25) & 0xFF == 27:
       data.apppend(cl)
    else:
        data.append(lv)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    
train_model(data)
