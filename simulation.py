#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from visual_simulation_class import Visual_simulator
from RL_algorithms import Reinforcement_Learning_algoritmhs

if __name__ == '__main__':
    
    # var names in all CAPS are meant not to be changed ever
    # coordinates counting from up to down, left to right in the grid:
    GOAL = (0, 3)
    SHIPWRECK = (2, 2)
    CRACKS = [(1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)]
    TERMINALS = set([GOAL] + CRACKS)
    # S_PLUS -> {(0, 0), .., (3, 3)}
    S_PLUS = set([(i, j) for i in range(4) for j in range(4)])
    S = S_PLUS - TERMINALS
    
    # only non-terminal states are listed. Example: A[(0, 0)] -> ['D', 'R']
    A = {  
         (0, 0): ['D', 'R'],
         (0, 1): ['D', 'L', 'R'],
         (0, 2): ['D', 'L', 'R'],
         (1, 0): ['U', 'D', 'R'],
         (1, 2): ['U', 'D', 'L', 'R'],
         (2, 0): ['U', 'D', 'R'],
         (2, 1): ['U', 'D', 'L', 'R'],
         (2, 2): ['U', 'D', 'L', 'R'],
         (3, 0): ['U', 'R']
         }
    
    # example: up is one row up (-1) and no column change (0)
    ACTION_VEC = {  
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1)
            }
    ACTION_TO_INT = {
            'U': 0,
            'D': 1,
            'L': 2,
            'R': 3
        }
    
    
    # Initialize RL algorithms class
    RL = Reinforcement_Learning_algoritmhs(S, A, S_PLUS, ACTION_VEC, \
                                          ACTION_TO_INT, TERMINALS, SHIPWRECK, \
                                          GOAL, CRACKS)
    
    V_list = RL.random_policy()
    
    DIMENSIONS = (4,4)
    GOAL = [(0, 3)]
    WITH_REWARD = [(2, 2)]
    NO_REWARD = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2),(3,0)]
    START = [(3,0)]
    
    # Initialize Visualization class
    P = RL.get_P()
    Viz = Visual_simulator(DIMENSIONS, CRACKS, NO_REWARD, WITH_REWARD, GOAL, START, P)
    Viz.show_simulation(V_list)
