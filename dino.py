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

import tools

class Game():
    
    def __init__(self, selenium=True):
        self.selenium = selenium
        driver_path = '/Users/gregoire/Documents/git/ChromeDino/chromedriver'
        if self.selenium:
            chrome_options = Options()
            #chrome_options.add_argument('--start-fullscreen')
            self.driver = webdriver.Chrome(driver_path, options=chrome_options)
            self.driver.get('https://chromedino.com/')
            self.body = self.driver.find_element_by_css_selector('body')
        
        # Starts as soon as the page is fully loaded
        
        # Define the size of the mask
        self.h_mask = 130
        self.w_mask = 600
        
        self.vision = Vision(self.h_mask, self.w_mask)
        
        self.game_over=False
        
        t = time.time()
        self.vision.get_position_roi()
        print('Get position ROI: ', time.time()-t)
        self.roi = self.vision.grab_roi() # Get screenshot of initial state
        
        cv2.imshow('Game', self.roi)
        
        self.start_game()
        self.play()
    
    def jump(self, action='up'):
        if action is not None:
            if self.selenium:
                if action == 'up':
                    self.body.send_keys(Keys.ARROW_UP)
                    print('up')
                else:
                    self.body.send_keys(Keys.ARROW_DOWN)
                    print('down')
            else:
                pyautogui.press(action)
                print('Not selenium', action)
                
    def start_game(self):
        print('>> Start')
        self.jump()
        time.sleep(1)
    
    def play(self):
        while not self.game_over:
            self.get_action()
            
            if self.is_game_over():
                self.game_over = True
                break
                
        print(">> Game Over!")
        self.driver.quit()
        cv2.destroyAllWindows()


    
    def get_action(self):
        self.roi = self.vision.grab_roi()
        
        p = random()
        print('Proba: ', p)
        if p>0.5:
            self.jump('up')
    
    def is_game_over(self):
        screen1 = self.vision.grab_roi()
        time.sleep(0.1)
        screen2 = self.vision.grab_roi()
        
        return np.array_equal(screen1, screen2)

        
        
class Vision():
    
    def __init__(self, h_mask, w_mask):
        self.h_mask = h_mask
        self.w_mask = w_mask
        self.window_name = 'Game'
        
        cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window_name, 0, 0)
        
    def get_position_roi(self):
        dino = np.asarray(Image.open(r"images/dino.png"))
        screenshot  = np.asarray(ImageGrab.grab())
        
        img = screenshot.copy()
        template = dino.copy()
        method = cv2.TM_CCORR

        w, h, _ = template.shape[::-1]
        
        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        self.bottom_left_roi = (top_left[0], top_left[1] + h)
    
    def grab_roi(self):
        bottom_left_roi_x, bottom_left_roi_y = self.bottom_left_roi
        
        roi = tools.screen_capture(bottom_left_roi_y - self.h_mask, bottom_left_roi_x, self.w_mask, self.h_mask)
        
        # Displays what the algorithm sees
        cv2.imshow(self.window_name, roi)
        cv2.waitKey(1)
        
        return roi
    
        
if __name__ == '__main__':
    game = Game()
