#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygame as pg
import numpy as np
import random
import matplotlib.pyplot as plt
import time

'''
Class created to allow a better visualization of the algorithms 
and to check the transition probability function 
'''

class Visual_simulator:

    # DIMENSIONS
    # tuple with window dimensions (i.e. number of squares in each side)
    # e.g. (3,4) --> 3 rows, 4 columns grid

    # TERMINAL STATES
    # list of tuples of nxn grid coordinates 
    # e.g. goal_state = [(0,0)  --> top left corner 
    #                    (0,n)  --> top rigth corner
    #                    (nxn)] --> bottom right corner

    # NON TERMINAL STATES (ZERO_REWARD + REWARD)
    # list of tuples of coordinates including reward 
    # e.g. [(1,2,10)] --> coordinate (1,2) reward 10
    def __init__(self,dimensions,terminal_states,zero_reward_states,reward_states,\
                 goal_state,start_state,trans_probs,\
                 load_images=['robot.png','crack.png','ship.png']):

        self.dimensions = dimensions
        self.terminal_states = terminal_states
        self.zero_reward_states = zero_reward_states
        self.reward_states = reward_states
        self.goal_state = goal_state
        self.start_state = start_state
        self.non_terminal_states = zero_reward_states 
        self.BLACK = (0,0,0) # color terminal states
        self.WHITE = (255,255,255) # color non-terminal states
        self.GREEN = (0,255,0) # color goal state
        self.RED = (255,0,0) # color start (current) state
        self.path_image_robot = load_images[0]
        self.path_image_crack = load_images[1]
        self.path_image_ship = load_images[2]
        self.trans_probs = trans_probs
        
        
    # count number of states to split the grid
    def count_states(self):
        num_states = len(self.t_states)+1
        return num_states
    
    # calculate dimensions in pixel for the game window
    def calculate_window_resolution(self):
        height = 100*self.dimensions[1]
        width = 100*self.dimensions[0]
        return height, width
    
    def draw_terminal_states(self,window, picture=None):
        for state in self.terminal_states: 
            if picture is None:
                pg.draw.rect(window, self.BLACK, (state[1]*100,state[0]*100,100,100))
            else:
                window.blit(picture, (state[1]*100,state[0]*100))
            pg.display.update()
        return

    def draw_goal_state(self,window):
        for state in self.goal_state:
            pg.draw.rect(window, self.GREEN, (state[1]*100,state[0]*100,100,100))
            pg.display.update()
        return
    
    def draw_reward_states(self,window,picture=None):
        for state in self.reward_states:
            if picture is None:
                pg.draw.rect(window, self.GREEN, (state[1]*100,state[0]*100,100,100))
            else:
                window.blit(picture,(state[1]*100,state[0]*100,100,100))
            pg.display.update()
        return

    def draw_start_state(self,window):
        for state in self.start_state:
            pg.draw.rect(window, self.RED, (state[1]*100,state[0]*100,100,100))
            pg.display.update()
        return

    def draw_current_state(self,window, current_state, picture=None):
        if picture is None:
            pg.draw.rect(window, self.RED, (current_state[1]*100,current_state[0]*100,100,100))
        else:
            window.blit(picture, (current_state[1]*100,current_state[0]*100))
        pg.display.update()
        return
    
    # used also to update all non_terminal states 
    # the distinction between reward and zero_reward is useful only
    # to print the ship picture during the manual playing simulator
    def draw_zero_reward_states(self,window,color=None):
        
        if color is None:
            color = self.WHITE
        
        for state in self.zero_reward_states:
            # points to define the sub-triangles of each square
            top_l_point = (state[1]*100,state[0]*100)
            top_r_point = (state[1]*100,state[0]*100+100)
            middle_point = (state[1]*100+50,state[0]*100+50)
            bottom_r_point = (state[1]*100+100,state[0]*100+100)
            bottom_l_point = (state[1]*100+100,state[0]*100)

            # sub-triangles coordinates 
            top_triangle = [top_l_point,top_r_point,middle_point]
            left_triangle = [top_l_point,bottom_l_point,middle_point]
            right_triangle = [top_r_point,bottom_r_point,middle_point]
            bottom_triangle = [bottom_r_point,bottom_l_point,middle_point]

            # draw triangles of each state
            pg.draw.polygon(window, color, top_triangle)
            pg.draw.polygon(window, color, right_triangle)
            pg.draw.polygon(window, color, left_triangle)
            pg.draw.polygon(window, color, bottom_triangle)

            pg.display.update()
        return
    
    def draw_non_terminal_states(self,window,values=None,colors=None):
        
        font = pg.font.SysFont('Arial', 25)
        
        for state in self.non_terminal_states:
            # points to define the sub-triangles of each square
            top_l_point = (state[1]*100,state[0]*100)
            top_r_point = (state[1]*100,state[0]*100+100)
            middle_point = (state[1]*100+50,state[0]*100+50)
            bottom_r_point = (state[1]*100+100,state[0]*100+100)
            bottom_l_point = (state[1]*100+100,state[0]*100)

            # sub-triangles coordinates 
            top_triangle = [top_l_point,top_r_point,middle_point]
            left_triangle = [top_l_point,bottom_l_point,middle_point]
            right_triangle = [top_r_point,bottom_r_point,middle_point]
            bottom_triangle = [bottom_r_point,bottom_l_point,middle_point]

            # draw triangles of each state
            if colors is None:
                pg.draw.polygon(window, self.WHITE, top_triangle)
                pg.draw.polygon(window, self.WHITE, right_triangle)
                pg.draw.polygon(window, self.WHITE, left_triangle)
                pg.draw.polygon(window, self.WHITE, bottom_triangle)
            else:
                pg.draw.polygon(window, colors[state[0],state[1]], top_triangle)
                pg.draw.polygon(window, colors[state[0],state[1]], right_triangle)
                pg.draw.polygon(window, colors[state[0],state[1]], left_triangle)
                pg.draw.polygon(window, colors[state[0],state[1]], bottom_triangle)
            
            if (values is not None) and (values[state[0],state[1]] < 20) :
                window.blit(font.render("%.2f" %values[state[0],state[1]], \
                                        True, self.BLACK), top_l_point)
            else:
                window.blit(font.render("%.2f" %values[state[0],state[1]], \
                                        True, self.WHITE), top_l_point)
                
            pg.display.update()
            
        return

    # load picture of robot and cracks instead of drawing squares
    def load_picture(self, path_image):
        picture = pg.image.load(path_image)
        return picture
    
    # get coordinates new state after single action
    def coordinates_next_state(self, current_state, action):

        next_state_trans_probs = self.trans_probs[current_state[0],\
                                                  current_state[1],action] 
        # non-zero indices from the transition matrix (i.e. allowed future states)
        indices = np.nonzero(next_state_trans_probs)
        
        # no possible movements (e.g. try to get over the edge of the grid)
        if len(indices[0]) == 0:
            return current_state
        
        # close to the edge (i.e. even if I slip I can make only one step)
        elif len(indices[0]) == 1:
            new_state = []
            new_state.append(indices[0][0])
            new_state.append(indices[1][0])
            return tuple(new_state)
        
        # position in which we can slip
        else:
            random_value = random.uniform(0,1)

            # there are only two possible probabilities
            if random_value < 0.05:
                new_indices = np.where(next_state_trans_probs==0.05)
            else:
                new_indices = np.where(next_state_trans_probs==0.95)
                
            new_state = []
            new_state.append(new_indices[0][0])
            new_state.append(new_indices[1][0])

            return tuple(new_state)
        
    # get rgb colors from matplotlib colormap
    def get_rbg_from_colormap(self, colormap):
        
        rgb_matrix = np.empty((colormap.shape[0],colormap.shape[1]),dtype='O')
        
        # compress third dimensions of colormap matrix to get a 2D matrix with rbg tuples
        for raw in range(colormap.shape[0]):
            for col in range(colormap.shape[1]):
                triple = []
                triple.append(colormap[raw,col,0]*255)
                triple.append(colormap[raw,col,1]*255)
                triple.append(colormap[raw,col,2]*255)
                rgb_matrix[raw,col] = tuple(triple)

        return rgb_matrix
    
    def min_max_scaler(self,matrix):
        matrix = (matrix-matrix.min())/(matrix.max()-matrix.min())
        return matrix
    
    # initialize the enviorment and let to play with it manually 
    def manual_simulation(self):
         
        pg.init()
        current_state = self.start_state[0]

        # start window
        height, width = self.calculate_window_resolution()
        window = pg.display.set_mode((width,height))
        
        if self.path_image_robot is not None:
            path_robot = self.path_image_robot
            robot_pic = self.load_picture(path_robot)
        if self.path_image_crack is not None:
            path_crack= self.path_image_crack
            crack_pic = self.load_picture(path_crack)
        if self.path_image_ship is not None:
            path_ship = self.path_image_ship
            ship_pic = self.load_picture(path_ship)
        
        simulation = True
        
        # start simulation
        while simulation:

            # fill window with states
            self.draw_terminal_states(window, crack_pic)
            self.draw_goal_state(window)
            self.draw_zero_reward_states(window)
            self.draw_reward_states(window, ship_pic)
            self.draw_current_state(window, current_state, robot_pic)

            for event in pg.event.get():

                # to allow exit window by pressing x
                if event.type == pg.QUIT:
                    simulation = False
                
                if event.type== pg.KEYDOWN:
                    
                    if event.key == pg.K_LEFT:
                        current_state = tuple(self.coordinates_next_state(current_state,2))
    
                    elif event.key == pg.K_RIGHT:
                        current_state = tuple(self.coordinates_next_state(current_state,3))
                        
                    elif event.key == pg.K_DOWN:
                        current_state = tuple(self.coordinates_next_state(current_state,1))
                        
                    elif event.key == pg.K_UP:
                        current_state = tuple(self.coordinates_next_state(current_state,0))
                
                if (current_state in self.terminal_states) or (current_state in self.goal_state):
                    current_state = self.start_state[0]
        return
    
    
    # visualize the changing of values during a training using colormaps
    # save the final optimal policy eventually
    def show_simulation(self, V_list):
        
        # define colormap 
        colormap = plt.get_cmap('Blues') #('Wistia')
        #min_max_scaler = preprocessing.MinMaxScaler()
        
        # load crack picture 
        if self.path_image_crack is not None:
            path_crack= self.path_image_crack
            crack_pic = self.load_picture(path_crack)
            
        # start pygame
        pg.init()
        
        # start window
        height, width = self.calculate_window_resolution()
        window = pg.display.set_mode((width,height))
        
        # start simulation
        simulation = True
        idx = 0
        while simulation:

            try:
                V = self.min_max_scaler(V_list[idx])
                colors = colormap(V)
                print(colors[0,0], V_list[idx][0,0],V[0,0])
                print(colors[1,2], V_list[idx][1,2],V[1,2],'\n')
                rgb_colors = self.get_rbg_from_colormap(colors)
                
                # fill window with initial states
                self.draw_terminal_states(window, crack_pic)
                self.draw_goal_state(window)
                self.draw_non_terminal_states(window,values=V_list[idx]\
                                              ,colors=rgb_colors)
            except:
                pass
            
            idx+=1
            time.sleep(1)
            
            # to allow exit window by pressing x
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    simulation = False
                    
        return

