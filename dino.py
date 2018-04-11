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
    
    def __init__(self, training, window_name="Game", monitor=True):

        driver_path = '/Users/gregoire/Documents/git/ChromeDino/chromedriver'
        chrome_options = Options()
        #chrome_options.add_argument('--start-fullscreen')
        self.driver = webdriver.Chrome(driver_path, options=chrome_options)
        self.driver.get('https://chromedino.com/')
        self.body = self.driver.find_element_by_css_selector('body')

        ## PONG
        self.training = training
        self.episode_hidden_layer_values, self.episode_observations, self.episode_gradient_log_ps, self.episode_rewards = [], [], [], []
        self.reward_sum = 0
        
        ####
        # Starts as soon as the page is fully loaded
        ####
        
        # Define the size of the mask
        self.h_mask = 100
        self.w_mask = 600
                
        # Launch recording and Vision system
        self.vision = Vision(self.h_mask, self.w_mask, window_name, monitor)        
        self.vision.get_position_roi()
                
        # Launch game   
        self.number_jumps = 0
        self.game_over=False
        self.start_game()
        self.play()
    
    
    def jump(self, action='up'):
        if action == 'up':
            self.body.send_keys(Keys.ARROW_UP)
            #print('up')
            self.number_jumps += 1
        elif action == 'down':
            self.body.send_keys(Keys.ARROW_DOWN)
            #print('down')
        else:
            pass
          
        
    def start_game(self):
        print('>> Start')
        self.jump() # Jump to begin playing
        time.sleep(1)
        self.time_start = time.time() # Needs to be after time.sleep(1)

        
    def play(self):
        self.prev_processed_observations = None
        self.roi = self.vision.grab_roi()
        self.observation = self.roi
        self.previous_score = -45
        #time.sleep(3.5) # No jumps at the beginning
        
        while not self.game_over:
            self.get_action()
            
            if self.is_game_over():
                self.game_over = True
                break
        self.end_game()        
     
    
    def end_game(self):
        self.final_score = self.get_score()
        print('>> Game Over!')
        print('Final score: ', self.final_score)
        self.driver.quit()
        cv2.destroyAllWindows()
    
    
    def get_action(self):        
        # preprocess the observations
        self.processed_observations, self.prev_processed_observations = preprocess_observations(self.observation, self.prev_processed_observations, self.training.input_dimensions)

        # Sending the observations through our neural net to generate the probability of telling our AI to move up
        self.hidden_layer_values, self.up_probability = apply_neural_nets(self.processed_observations, self.training.weights)
        self.episode_observations.append(self.processed_observations)
        self.episode_hidden_layer_values.append(self.hidden_layer_values)

        # Choose an action
        self.action = choose_action(self.up_probability)
        
        # carry out the chosen action
        self.jump(self.action)
        ## FUSIONNER DEUX LIGNES CI-DESSOUS
        self.roi = self.vision.grab_roi()
        self.observation = self.roi
        
        self.reward = self.get_score() - self.previous_score
        self.previous_score = self.get_score()
        # Done: game over ou pas ?
        # info: useless ?
        # observation: on fait un screenshot
        # reward: pas besoin, on le temps ?

        self.episode_rewards.append(self.reward) # Différence de temps entre les deux instants ?
        
        # see here: http://cs231n.github.io/neural-networks-2/#losses
        self.fake_label = 1 if self.action == 'up' else 0
        self.loss_function_gradient = self.fake_label - self.up_probability
        self.episode_gradient_log_ps.append(self.loss_function_gradient)
        
    def is_game_over(self):
        screen1 = self.vision.grab_roi()
        time.sleep(0.1)
        screen2 = self.vision.grab_roi()
        
        return np.array_equal(screen1, screen2)
    
    
    def get_score(self):
        return int((time.time()-self.time_start)*10)- 45 - 5*self.number_jumps
        
        

class Vision():
    
    def __init__(self, h_mask, w_mask, window_name='Game', monitor=True):
        self.h_mask = h_mask
        self.w_mask = w_mask
        self.window_name = window_name
        self.monitor = monitor
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window_name, 0, 0)
        
        
    def get_position_roi(self):
        dino = np.asarray(Image.open(r"images/dino_full.png"))
        screenshot  = np.asarray(ImageGrab.grab())
        
        img = screenshot.copy()
        template = dino.copy()
        method = cv2.TM_CCOEFF
        
        h, w, _ = template.shape
        
        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        self.bottom_left_roi = (top_left[0], top_left[1] + h)
    
    
    def grab_roi(self):
        bottom_left_roi_x, bottom_left_roi_y = self.bottom_left_roi
        
        roi = tools.screen_capture(bottom_left_roi_y - self.h_mask, bottom_left_roi_x, self.w_mask, self.h_mask)
        
        edges = cv2.Canny(roi, 100, 200)
        
        # Displays what the algorithm sees
        if self.monitor:
            self.display_image(edges)
        
        return edges
    
    def display_image(self, image):
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)
    
        
if __name__ == '__main__':
    game = Game()
