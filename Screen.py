# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time

import cv2
import numpy as np
import pyautogui

#import tensorflow as tf

e = 0

from PIL import ImageGrab

#from directKeys import PressKey, ReleaseKey , W, A, S ,D


##################################
#drawing lines function
##################################

def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0]),(coords[1]),(coords[2]),(coords[3]),[255,255,255], 3)
    except:
        pass

##################################
#region of interest
##################################

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img,mask)
    ##return masked
    return img



##################################
#precessing images
##################################



def process_img(original_image):
    ##gets the image and gray scales it 
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=500)
    
    ##adding a blur
    processed_img = cv2.GaussianBlur(processed_img, ( 5, 5), 0)
    
    vertices = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])
    ##asking area of screen not needed
    processed_img = roi(processed_img, [vertices])
    
    ##getting lines for logic
    ##processed_img needs to be edges - which is the canny edges
    lines =  cv2.HoughLinesP(processed_img, 1, np.pi/180, 180,np.array([]), 150, 50)
    draw_lines(processed_img,lines)
    
    
    return processed_img



##################################
#count down to have toime to get 
#to the game
##################################

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)




##################################
#getting the images
##################################

#
#mon = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
#with mss() as sct:
#    # mon = sct.monitors[0]
#    while True:
#        last_time = time.time()
#        img = sct.grab(mon)
#        print('fps: {0}'.format(1 / (time.time()-last_time)))
#        cv2.imshow('test', np.array(img))
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break



   
def e():
    last_time = time.time()
    while(True):
        screen = np.array( ImageGrab.grab(bbox=(0,40,800,640)))
            
        new_screen = process_img(screen)

        
        
    #   printscreen_numpy = np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1],printscreen_pil.size[0],3))
            
        
            
        #print('Loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time();
        cv2.imshow('window', new_screen)
        cv2.imshow('window2', cv2.cvtColor( screen, cv2.COLOR_BGR2RGB))
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


