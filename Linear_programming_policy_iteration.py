# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:48:49 2019

@author: Joop
"""

import numpy as np
from scipy.optimize import linprog

# var names in all CAPS are supposed to be constants
# coordinates counting from up to down, left to right in the grid:
GOAL = (0, 3)
SHIPWRECK = (2, 2)
CRACKS = [(1, 1), (1, 3), (2, 3), (3, 1), (3, 2), (3, 3)]
TERMINALS = set([GOAL] + CRACKS)
# S_PLUS -> {(0, 0), .., (3, 3)}
S_PLUS = set([(i, j) for i in range(4) for j in range(4)])
S = S_PLUS - TERMINALS
A = {  # only non-terminal states are listed. Example: A[(0, 0)] -> ['D', 'R']
     (0, 0): ['U', 'D', 'L', 'R'],
     (0, 1): ['U', 'D', 'L', 'R'],
     (0, 2): ['U', 'D', 'L', 'R'],
     (1, 0): ['U', 'D', 'L', 'R'],
     (1, 2): ['U', 'D', 'L', 'R'],
     (2, 0): ['U', 'D', 'L', 'R'],
     (2, 1): ['U', 'D', 'L', 'R'],
     (2, 2): ['U', 'D', 'L', 'R'],
     (3, 0): ['U', 'D', 'L', 'R']
     }
ACTION_VEC = {  # example: up is one row up (-1) and no column change (0)
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
        }
ACTION_TO_INT = {  # integer representation of an action. Helper function
        'U': 0,
        'D': 1,
        'L': 2,
        'R': 3
        }
GAMMA = .9


# reward doesn't depend on s or a, just on s'
def R(s, a, s_):
    if s_ == GOAL:
        r = 100
    elif s_ == SHIPWRECK:
        r = 20
    elif s_ in CRACKS:
        r = -10
    else:
        r = 0
    return r


# transition probabilities
def get_P():
    # initialize array with 4 * 4 * s, 4 * a, and 4 * 4 * s_ to all zeros
    P = np.zeros((4, 4, 4, 4, 4), dtype=np.float16)
    for s in S:
        for a in A[s]:
            # define state s_new that agent intends to go to
            s_new = (s[0] + ACTION_VEC[a][0], s[1] + ACTION_VEC[a][1])
            for s_ in S_PLUS:
                if s_new == s_:
                    if s_ in TERMINALS:
                        p = 1  # no sliding over terminals
                    elif a == 'U' and s_[0] == 0:
                        p = 1  # no sliding up over top row states
                    elif a == 'D' and s_[0] == 3:
                        p = 1  # no sliding down over bottom row states
                    elif a == 'L' and s_[1] == 0:
                        p = 1  # no sliding left over left column states
                    else:  # this s' has possibility of sliding over
                        p = .95  # '''0.95'''
                        # define sliding end states for assigning P = .05
                        if a == 'U':
                            s_slip = (0, s_[1])  # first row, column of s_
                        elif a == 'D':
                            s_slip = (3, s_[1])  # last row, column of s_
                        elif a == 'R':
                            s_slip = (s_[0], 3)  # row of s_, last column
                        elif a == 'L':
                            s_slip = (s_[0], 0)  # row of s_, first column
                        P[s[0], s[1], ACTION_TO_INT[a], s_slip[0], s_slip[1]]\
                            = .05 #'''0.05'''
                    P[s[0], s[1], ACTION_TO_INT[a], s_[0], s_[1]] = p
                else:  # we leave p to 0
                    pass
    return P

def get_A_ub():
    A_ub = []
    for s in S:
        for a in A[s]:
            A_ub_el = np.empty(len(S), dtype=np.float16)
            E_R_sa = 0
            for s_ in S_PLUS:
                E_R_sa += P[(*s, ACTION_TO_INT[a], *s_)] * R(None, None, s_)
            b_ub.append(-E_R_sa)
            for j, s_ in enumerate(S):
                if s == s_:
                    A_ub_el[j] = -1
                else:
                    A_ub_el[j] = GAMMA * P[(*s, ACTION_TO_INT[a], *s_)]
            A_ub.append(A_ub_el)
    return np.array(A_ub)


def get_V():
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  options={"disp": True})
    print(res, '\n')
    print(S, res.x)
    V = np.zeros((4, 4), dtype=np.float16)
    for i, s in enumerate(S):
        V[s] = res.x[i]
    print(V, '\n')
    return V


def pol_improv(V):
    for s in S:
        Q = {}
        for a in A[s]:
            Q[a] = 0
            for s_ in S_PLUS:
                Q[a] += P[(*s, ACTION_TO_INT[a], *s_)] *\
                    (R(None, None, s_) + GAMMA * V[s_])
        pi[s] = max(Q, key=Q.get)


P = get_P()
S = list(S)

# first we get V from linear programming, then one round of policy improvement
c = np.ones(len(S))
bounds = (None, None)
b_ub = []
A_ub = get_A_ub()
V = get_V()
pi = {  # start with arbitrary pi(s)
      (0, 0): 'D',
      (0, 1): 'D',
      (0, 2): 'D',
      (1, 0): 'U',
      (1, 2): 'U',
      (2, 0): 'U',
      (2, 1): 'U',
      (2, 2): 'U',
      (3, 0): 'U'
      }


pol_improv(V)
print('Optimal policy:', '\n', pi)
