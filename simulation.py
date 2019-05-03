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
    S_PLUS = set([(i, j) for i in range(4) for j in range(4)]) # coordinates of the whole grid
    S = S_PLUS - TERMINALS
    
    # only non-terminal states are listed.
    A = {
        (0, 2): ['U', 'D', 'L', 'R'],
        (0, 1): ['U', 'D', 'L', 'R'],
        (1, 2): ['U', 'D', 'L', 'R'],
        (0, 0): ['U', 'D', 'L', 'R'],
        (2, 2): ['U', 'D', 'L', 'R'],
        (1, 0): ['U', 'D', 'L', 'R'],
        (2, 1): ['U', 'D', 'L', 'R'],
        (2, 0): ['U', 'D', 'L', 'R'],
        (3, 0): ['U', 'D', 'L', 'R']
        }
    
    # example: up is one row up (-1) and no column change (0)
    ACTION_VEC = {  
            'L': (0, -1),
            'R': (0, 1),
            'U': (-1, 0),
            'D': (1, 0)
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
    
    '''Random policy'''
#    V_list, iterations = RL.random_policy_evaluation()
#    print(iterations)
    
    '''Value iteration'''
    V_list, iterations, policies = RL.values_iteration()
    print('Iterations: ', iterations)
    print(V_list)
#    print(policies[-1])
    
    '''Howard's Policy Iteration'''
    
     # start with arbitrary pi(s)
#    pi = { 
#            (0, 2): 'D',
#            (0, 1): 'D',
#            (1, 2): 'U',
#            (0, 0): 'D',
#            (2, 2): 'U',
#            (1, 0): 'U',
#            (2, 1): 'U',
#            (2, 0): 'U',
#            (3, 0): 'U'
#        }
#    V_list = RL.Howard_policy_iteration(pi)
#    print(V_list)
    
    # need to redefine some element for the visualization class
    DIMENSIONS = (4,4)
    GOAL = [(0, 3)]
    WITH_REWARD = [(2, 2)]
    NO_REWARD = [(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2),(3,0)]
    START = [(3,0)]
    
    '''Visualization class'''
    #Initialize Visualization class
#    P = RL.get_P()
#    Viz = Visual_simulator(DIMENSIONS, CRACKS, NO_REWARD, WITH_REWARD, GOAL, START, P)
#    Viz.show_simulation(V_list)
    
    '''Function for manual evaluation of the enviornemnt'''
    #Viz.manual_simulation()
    

