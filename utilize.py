import numpy as np
from scipy.special import expit


def true_q_function_drl(state, action):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    q_value = 0.533899 - 0.0122839 * action1 + 0.457572 * state1 - 0.27498 * action1 * state1
    q_value = 10 * q_value
    return q_value

def false_q_function_drl(state, action):
    true_q = true_q_function_drl(state, action)
    noise_q = true_q + np.random.normal(loc=1, scale=1, size=true_q.shape[0])
    return noise_q

def true_q_function(state, action, mediator):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    q_value = 5.45503 + 0.0366013*mediator1 + 0.0534233*action1*mediator1 + 0.0100351*state1 + \
        0.0539304 * state1*action1 - 0.00404461 * state1 * mediator1 - 0.00446658*action1*mediator1*state1
    q_value = 10 * q_value
    return q_value

def false_q_function(state, action, mediator):
    true_q = true_q_function(state, action, mediator)
    noise_q = true_q + np.random.uniform(low=0, high=1, size=true_q.shape[0])
    return noise_q

def true_pa_function(state, action):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    pa_one = 0.5 + 0.02054205*state1
    pa = action1*pa_one + (1.0-action1)*(1.0-pa_one)
    return pa

def false_pa_function(state, action):
    true_pa = true_pa_function(state, action)
    noise_pa = true_pa * np.random.uniform(low=0.75, high=1.0, size=true_pa.shape[0])
    noise_pa = np.clip(noise_pa, a_min=0.01, a_max=1.0)
    return noise_pa

def true_pm_function(state, action, mediator):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    pm_one = expit(0.1*state1 + 0.9*action1 - 0.45)
    pm = mediator1*pm_one + (1.0-mediator1)*(1.0-pm_one)
    return pm

def false_pm_function(state, action, mediator):
    true_pm = true_pm_function(state, action, mediator)
    noise_pm = true_pm * np.random.uniform(low=0.7, high=1.2, size=true_pm.shape[0])
    noise_pm = np.clip(noise_pm, a_min=0.01, a_max=1.0)
    return noise_pm

def true_ratio_function(state):
    ## the ratio value is computed from the "verify_ratio.py"
    # ratio = np.array([1.02712501, 0.98431272])
    ratio = np.array([1.03375778, 0.97257431])
    state1 = np.copy(state).flatten()
    state1 = state1.astype(int)
    return ratio[state1]

def false_ratio_function(state):
    ratio = np.array([1.03375778, 0.97257431])
    noise_ratio = np.random.uniform(low=0.0, high=1.0, size=2)
    ratio = ratio + noise_ratio
    state1 = np.copy(state).flatten()
    state1 = state1.astype(int)
    return ratio[state1]
