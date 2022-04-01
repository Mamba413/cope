# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit

def target_policy(state, action=None):
    pa = 0.3*np.sum(state)
    pa = expit(pa)
    if pa.ndim == 1:
        pa = pa[0]
    elif pa.ndim == 2:
        pa = pa[0][0]
        pass

    prob_arr = np.array([1-pa, pa])
    
    if action is None:
        action_value = np.random.choice([0, 1], 1, p=prob_arr)
    else:
        action_value = np.array([prob_arr[int(action)]])
    return action_value

def target_policy_action3(state, action=None):
    pa = 0.3*np.sum(state)
    pa = expit(pa)
    if pa.ndim == 1:
        pa = pa[0]
    elif pa.ndim == 2:
        pa = pa[0][0]
        pass

    half_pa = 0.5 * pa
    prob_arr = np.array([half_pa, 1 - 2 * half_pa, half_pa])
    
    if action is None:
        action_value = np.random.choice([-1, 0, 1], 1, p=prob_arr)
    else:
        action_value = np.array([prob_arr[int(action) + 1]])
    return action_value

def target_policy_action3_inf(state, action=None):
    pa = 0.3*np.sum(state)
    pa = expit(pa)
    if pa.ndim == 1:
        pa = pa[0]
    elif pa.ndim == 2:
        pa = pa[0][0]
        pass

    prob_arr = np.array([0.25 * pa, 1 - pa, 0.75 * pa])
    
    if action is None:
        action_value = np.random.choice([-1, 0, 1], 1, p=prob_arr)
    else:
        action_value = np.array([prob_arr[int(action) + 1]])
    return action_value
