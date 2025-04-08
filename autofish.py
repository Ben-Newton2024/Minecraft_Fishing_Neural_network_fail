# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 16:43:39 2021

@author: bened
"""

import pyautogui
import cv2
from PIL import ImageGrab
from time import sleep
import numpy as np


def initializePyAutoGUI():
    # Initialized PyAutoGUI
    # When fail-safe mode is True
    # moving the mouse to the upper-left
    # corner will abort your program. This prevents 
    # locking the program up.
    pyautogui.FAILSAFE = True


def take_capture(magnification):
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
    return res


def autofish(tick_interval, threshold, magnification):
    pyautogui.rightClick()  # cast the fishing line
    print('cast line')
    sleep(2)  # wait a couple of seconds before taking captures
    img = take_capture(magnification)  # take initial capture 

    # Continue looping to take a capture and convert and check 
    # until there are no black pixels in the capture. This will 
    # display the image, but it isn't necessary (the imshow method).
    # Once there are no black pixels in the capture:
    #     np.sum(img == 0) is looking for black pixels
    #     > threshold is the number of those pixels (0) 
    # exit the loop and reel in the catch (pyautogui.rightClick()).
    # Finally, wait a second and leave the auto-fish method.
    # This will cast, wait and catch one interval. See main method 
    # for looping.
    while np.sum(img == 0) > threshold:
        img = take_capture(magnification)
        sleep(tick_interval)
        cv2.imshow('window', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit
            break
    pyautogui.rightClick()
    print('real line: no black pixels')
    sleep(5)


# This will wait 5 seconds to allow switching from Python program
# to Minecraft. Then loop through the autofish method for 100 
# cast and catch loops.
# 
# Launch Minecraft and load up your world
# Equip your fishing pole and be ready to cast into a fishable area
# Run program through IDLE or your IDE
# Switch to the Minecraft while running
# Position character so that it is ready to cast 
# and the cursor will be immediately on top of the bobber 
# Let it run...
# If you need more time, change sleep(5) to something more
def main():
    initializePyAutoGUI()
    sleep(5)
    while True:
        autofish(0.01, 0, 5)


if __name__ == "__main__":
    main()
