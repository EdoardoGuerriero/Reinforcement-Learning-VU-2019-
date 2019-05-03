#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import operator 

'''
Class containing the algorithms required for the assignment 
'''

class Reinforcement_Learning_algoritmhs:
    
    def __init__(self, S,A,S_PLUS,ACTION_VEC,ACTION_TO_INT,TERMINALS,SHIPWRECK,GOAL,CRACKS):
        
        self.S = S  # non terminal states
        self.A = A  # actions for each state
        self.S_PLUS = S_PLUS # set of all states of the grid
        self.ACTION_VEC = ACTION_VEC # vector to sum to perform a movement 
        self.ACTION_TO_INT = ACTION_TO_INT # number code for each action
        self.TERMINALS = TERMINALS # terminal states
        self.SHIPWRECK = SHIPWRECK # states with reward
        self.GOAL = GOAL # goal state
        self.CRACKS = CRACKS# terminal states
        self.P = self.get_P()
        
    # uniform probabilities for random policy iteration
    def pi_uniform(self, s, a):
        if a in self.A[s]:
            p = 1 / len(self.A[s])  # random policy
        else:
            p = 0
        return p
    
    
    def get_reward(self,s, a,s_):
        if s_ == self.GOAL:
            r = 100
        elif s_ == self.SHIPWRECK:
            r = 20
        elif s_ in self.CRACKS:
            r = -10
        else:
            r = 0
        return r
        
    # Function to get transition probabilities 
    def get_P(self):
        # initialize array with 4 * 4 * s, 4 * a, and 4 * 4 * s_ to all zeros
        P = np.zeros((4, 4, 4, 4, 4), dtype=np.float64)
        for s in self.S:
            
            for a in self.A[s]:
                # define state s_new that agent intends to go to
                s_new_ = (s[0] + self.ACTION_VEC[a][0], s[1] + self.ACTION_VEC[a][1])
                
                if s_new_ in self.S_PLUS:
                    s_new = s_new_
                else:
                    s_new = s
                #print('s ', s, ' a ', a, ' new_s ', s_new)
                for s_ in self.S_PLUS:
                    if s_new == s_:
                        
                        # no sliding over terminals
                        if s_ in self.TERMINALS:
                            p = 1  
                        
                        # no sliding up over top row states
                        elif a == 'U' and s_[0] == 0:
                            p = 1  
                        
                        # no sliding down over bottom row states
                        elif a == 'D' and s_[0] == 3:
                            p = 1  
                        
                        # no sliding left over left column states
                        elif a == 'L' and s_[1] == 0:
                            p = 1  
                            
                        else:  
                            # this s' has possibility of sliding over
                            p = .95

                            # define sliding end states for assigning P = .05
                            if a == 'U':
                                s_slip = (0, s_[1])  # first row, column of s_
                            elif a == 'D':
                                s_slip = (3, s_[1])  # last row, column of s_
                            elif a == 'R':
                                s_slip = (s_[0], 3)  # row of s_, last column
                            elif a == 'L':
                                s_slip = (s_[0], 0)  # row of s_, first column

                            P[s[0], s[1], self.ACTION_TO_INT[a], s_slip[0], s_slip[1]]\
                                = .05
                        P[s[0], s[1], self.ACTION_TO_INT[a], s_[0], s_[1]] = p

                    else:  # we leave p to 0
                        pass

        return P
    
    # Random Policy 
    def random_policy_evaluation(self,THETA=1e-6,GAMMA=0.9):
        
        P = self.get_P()
        V = np.zeros((4, 4), dtype=np.float64)  # grid values initialized to zeros
        delta = 1  # just to start the while loop
        V_list = []
        iterations = 0
        while delta > THETA:  # stop when delta < theta
            delta = 0
            for s in self.S:  # loop over non-terminal states only
                v = V[s]  # save old value
                V[s] = 0
                for a in self.A[s]:  # loop over allowed actions in the state
                    # accumulate utility for this action only
                    q = 0
                    for s_ in self.S_PLUS:
                        q += P[s[0], s[1], self.ACTION_TO_INT[a], s_[0], s_[1]] *\
                            (self.get_reward(None, None, s_) + GAMMA * V[s_])
                    V[s] += self.pi_uniform(s, a) * q
    
            #print(abs(v - V[s]))
            delta = max([delta, abs(v - V[s])])
            V_list.append(V.copy())
            iterations += 1
            
        return V_list, iterations
    
    
    # Value Iteration
    def values_iteration(self,THETA=1e-6,GAMMA=0.9):
        
        V = np.zeros((4, 4), dtype=np.float64)  # grid values initialized to zeros
        delta = 1  # just to start the while loop
        V_list = []
        policy_list = []
        V_list.append(V.copy()) # null initialization for visualization
        iterations = 0
        
        while delta > THETA:  # stop when delta < theta
        #for i in range(20):
            delta = 0
            policy = {}
            for s in self.S:  # loop over non-terminal states only
                v = V[s]  # save old value
                V[s] = 0
                Q = {'U': 0, 'D': 0, 'L': 0, 'R': 0} # list to store the estimated values for each action
                #Policy = []
                for a in self.A[s]:  # loop over allowed actions in the state
                    # accumulate utility for this action only
                    
                    for s_ in self.S_PLUS:
                        #print(s, ' ',s_)
                        # get transition probability for each point in the grid 
                        trans_P = self.P[s[0], s[1], self.ACTION_TO_INT[a], s_[0], s_[1]]
                        
                        # skip future states with trans_P = 0 (impossible to reach)
                        if trans_P != 0:
                            print(trans_P)
                            Q[a] = (trans_P*(self.get_reward(None, None, s_)\
                             +(GAMMA*V_list[iterations][s_])))
                            
                # select maximum utility value and correspondant action
                maxValue = max(Q.items(), key=operator.itemgetter(1))[1]
                V[s] += maxValue
                # store all actions with maximum value 
                policy[s] = [key for key in Q.keys() if Q[key]==maxValue]
                
                delta = max([delta, abs(v - V[s])])
#            print('v ', v, ' V[s] ', V[s])
#            print(delta)
            V_list.append(V.copy())
            policy_list.append(policy.copy())
            iterations += 1
            
        return V_list, iterations, policy_list
    
    def policy_evaluation(self, V, pi, THETA=1e-6, GAMMA=0.9):
        
        i = 0  # nr of repeats
        delta = 1  # just to start the while loop
        while delta > THETA:  # stop when delta < theta
            delta = 0
            for s in self.S:  # loop over non-terminal states only
                v = V[s]  # save old value
                V[s] = 0
                for s_ in self.S_PLUS:
                    V[s] += self.P[(*s, self.ACTION_TO_INT[pi[s]], *s_)] *\
                        (self.get_reward(None, None, s_) + GAMMA * V[s_])
                delta = max([delta, abs(v - V[s])])
            i += 1
        return V, i

    
    def policy_improvement(self, V, pi, GAMMA=0.9):
        
        stable = True
        
        for s in self.S:
            b = pi[s]
            Q = {}
            for a in self.A[s]:
                Q[a] = 0
                for s_ in self.S_PLUS:
                    Q[a] += self.P[(*s, self.ACTION_TO_INT[a], *s_)] *\
                        (self.get_reward(None, None, s_) + GAMMA * V[s_])
            pi[s] = max(Q, key=Q.get)
            if b != pi[s]:
                stable = False
                # with commenting out the line below, this becomes part 4 of assnmt
                # break  # in "simple pol it" we only update one action
        return stable
    
    
    def Howard_policy_iteration(self, pi):
        
        V = np.zeros((4, 4), dtype=np.float64)  # grid values initialized to zeros
        stable = False  # policy-stable
        eval_improv_list = []
        
        while not stable:
            V, i = self.policy_evaluation(V, pi)
            stable = self.policy_improvement(V, pi)
            eval_improv_list.append((V.copy(), i, pi.copy()))
            
        return eval_improv_list

    
    
    
    
    
    
