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

