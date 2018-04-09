#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import random
import pyautogui
import numpy as np
from PIL import Image, ImageGrab
import time
import cv2

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options


class Game():
    def __init__(self):
        driver_path = '/Users/gregoire/Documents/git/chromedriver'
        
        # Start selenium driver
        chrome_options = Options()
        chrome_options.add_argument('--start-fullscreen') # Doesn't work yet
        self.driver = webdriver.Chrome(driver_path, options=chrome_options)
        self.driver.get('https://chromedino.com/')
        self.body = self.driver.find_element_by_css_selector('body')
        
        # Starts as soon as the page is fully loaded
        
        # Define the size of the mask
        self.h_mask = 130
        self.w_mask = 600
        
        self.game_over=False
        
        # Retrieve coordinates of roi
        self.bbox_roi = self.get_bbox_roi()
        self.roi = self.grab_roi() # Get screenshot of initial state
        
        # Launch game and play
        self.start_game()
        self.play()
    
    def get_bbox_roi(self):
        '''
            Returns:
                tuple with the coordinates of the bounding box
                of the region of interest
        '''
        dino = np.asarray(Image.open(r"images/dino.png"))
        screenshot  = np.asarray(ImageGrab.grab())
    
        method = cv2.TM_CCORR
        w, h, _ = template.shape[::-1]

        # Apply template Matching
        res = cv2.matchTemplate(screenshot, dino, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Retrieve coordinates
        top_left = max_loc
        bottom_left_roi = (top_left[0], top_left[1] + h)
        bottom_left_roi_x, bottom_left_roi_y = bottom_left_roi
        
        return (bottom_left_roi_x, bottom_left_roi_y - self.h_mask,
                                    bottom_left_roi_x + self.w_mask, bottom_left_roi_y)
    
    def grab_roi(self):
        return ImageGrab.grab(bbox=self.bbox_roi)
    
    def jump(self, action='up'):
        if action is not None:
            if action == 'up':
                self.body.send_keys(Keys.ARROW_UP)
                print('up')
            else:
                self.body.send_keys(Keys.ARROW_DOWN)
                print('down')
                
    def start_game(self):
        self.jump()
    
    def play(self):
        while not self.game_over:
            self.get_action()
                
            t1 = np.array(ImageGrab.grab())
            t2 = np.array(ImageGrab.grab())

            if np.array_equal(t1, t2):
                self.game_over=True
                print("Game Over!")
    
    def get_action(self):
        if random()>0.5:
            self.jump()

