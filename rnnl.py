# -*- coding: utf-8 -*-
"""
@author: Jin Zhu
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class RatioRKHSLearner(tf.Module):
    def __init__(self, hidden_node, truncate=20):
        super(RatioRKHSLearner, self).__init__()
        self.hidden_layer = layers.Dense(hidden_node, activation='relu', dtype='float64')
        self.output_layer = layers.Dense(1, dtype='float64')
        self.truncate = truncate

    def __call__(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
    
    def get_ratio_prediction(self, state, normalize=True):
        if state.shape == ():
            state = state.reshape(1, 1)
        elif state.shape == (1, ):
            state = state.reshape(1, state.size)
        else:
            pass
        
        ratio = self.__call__(state)
        ratio = ratio.numpy()
        ratio = ratio.flatten()

        ## truncate:
        ratio_min = 1 / self.truncate
        ratio_max = self.truncate
        ratio = np.clip(ratio, a_min=ratio_min, a_max=ratio_max)
        if state.shape[0] > 1:
            if normalize:
                ratio /= np.mean(ratio)
        return ratio
    
    def get_r_prediction(self, state, normalize=True):
        return self.get_ratio_prediction(state, normalize)

def tf_gaussian_kernel(x, width):
    kernel_value = np.divide(np.exp(-x), 2.0*np.square(width))
    return kernel_value

def loss(model, s0, state, next_state, policy_ratio, width, gamma, batch_size):
    loss_term1_value = loss_term_1(model, s0, state, next_state, policy_ratio, width, gamma, batch_size)
    loss_term2_value = loss_term2(model, state, next_state, policy_ratio, width, gamma, batch_size)
    loss_term = loss_term1_value + loss_term2_value
    return loss_term

def term1_gap(s0, state, next_state, policy_ratio, width, gamma, batch_size): 
    gap_vec = np.zeros(batch_size)
    for i in range(batch_size):
        s0_state_dist = s0 - state[i, :]
        s0_state_dist = np.sqrt(np.sum(np.square(s0_state_dist), axis=1))
        kern_s0_s = tf_gaussian_kernel(s0_state_dist, width)
        s0_next_state_dist = s0 - next_state[i, :]
        s0_next_state_dist = np.sqrt(np.sum(np.square(s0_next_state_dist), axis=1))
        kern_s0_ns = tf_gaussian_kernel(s0_next_state_dist, width)
        gap = np.mean(kern_s0_s - gamma * policy_ratio[i] * kern_s0_ns)
        gap_vec[i] = gap
    return gap_vec

def loss_term_1(model, s0, state, next_state, policy_ratio, width, gamma, batch_size):
    weight_vec = term1_gap(s0, state, next_state, policy_ratio, width, gamma, batch_size)
    state_fit = model(state)
    term_1plus2_value = -2.0 * (1.0 - gamma) * tf.reduce_mean(state_fit * weight_vec)
    return term_1plus2_value

def term2_gap(state, next_state, policy_ratio, width, gamma):
    cross_ratio = np.square(gamma) * policy_ratio.reshape(-1, 1) * tf.transpose(policy_ratio)
    cross_kernel_ns = tf_gaussian_kernel(euclidean_distances(next_state), width)
    cross_kernel_s = tf_gaussian_kernel(euclidean_distances(state), width)
    cross_kernel_s_ns = tf_gaussian_kernel(euclidean_distances(state, next_state), width)
    cross_kernel_s_ns = cross_kernel_s_ns * policy_ratio
    gap_mat = cross_ratio * cross_kernel_ns + cross_kernel_s - 2.0 * gamma * cross_kernel_s_ns
    return gap_mat

def loss_term2(model, state, next_state, policy_ratio, width, gamma, batch_size):
    state_fit = model(state)
    cross_pred_mat = tf.reshape(state_fit, [-1, 1]) * tf.transpose(state_fit)
    gap_mat = term2_gap(state, next_state, policy_ratio, width, gamma)
    term3_value = tf.reduce_mean(cross_pred_mat * gap_mat)
    return term3_value

def gradient(model, s0, state, next_state, policy_ratio, width, gamma, batch_size):
    with tf.GradientTape() as tape:
        loss_value = loss(model, s0, state, next_state, policy_ratio, width, gamma, batch_size)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train_one_step(model, optimizer, s0, state, next_state, policy_ratio, width, gamma, batch_size):
    loss_value, grads = gradient(model, s0, state, next_state, policy_ratio, width, gamma, batch_size)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

def compute_pm_ratio(state, action, mediator, policy, cplearner, use_mediator=True):
    if use_mediator:
        unique_action = np.unique(action)
        ratio = np.zeros((action.shape[0], 1))
        for action_value in unique_action:
            action_value = np.array([action_value])
            target_pa = np.apply_along_axis(policy, 1, state, action=action_value).flatten()
            policy_action_tmp = np.repeat(action_value, action.shape[0]).reshape(-1, 1)
            pm_ratio = cplearner.get_pm_ratio(state, policy_action_tmp, action, mediator)
            ratio += (pm_ratio * target_pa).reshape(-1, 1)
        ratio = ratio.flatten()
    else:
        estimate_pa = cplearner.get_pa_prediction(state, action)
        target_pa = np.zeros([0.0])
        pa_ratio = target_pa / estimate_pa
        ratio = pa_ratio
    return ratio

def compute_median_distance(state, max_obs=500):
    if state.shape[0] <= max_obs:
        median_state_dist = np.median(euclidean_distances(state, state))
    else:
        sub_index = np.random.choice(state.shape[0], max_obs)
        state_part = state[sub_index, :]
        median_state_dist = np.median(euclidean_distances(state_part, state_part))
    return median_state_dist

def train(model, optimizer, dataset, policy, cplearner, gamma=0.9, batch_size=2048, epoch=1000, trace=True):
    s0 = dataset['s0']
    state = dataset['state']
    action = dataset['action']
    mediator = dataset['mediator']
    policy_ratio = compute_pm_ratio(state, action, mediator, policy, cplearner)
    next_state = dataset['next_state']
    median_state_dist = compute_median_distance(state)

    train_data_num = state.shape[0]
    train_loss_results = []
    batch_size = min(state.shape[0], batch_size)
    if trace:
        print("Start Training")
    for i in range(1, 1+epoch):
        np.random.seed(i)
        idx = np.random.permutation(train_data_num)
        state = state[idx, :]
        next_state = next_state[idx, :]
        policy_ratio = policy_ratio[idx]

        for k in range(int(train_data_num/batch_size)):
            s = state[k*batch_size:(k+1)*batch_size, :]
            ns = next_state[k*batch_size:(k+1)*batch_size, :]
            pr = policy_ratio[k*batch_size:(k+1)*batch_size]
            loss_value = train_one_step(model, optimizer, s0, s, ns, pr, median_state_dist, gamma, batch_size)
            # print(loss_value)
            # print(loss_value.shape)
        
        train_loss_results.append(loss_value)
        if trace and (i % 5 == 0):
            print("Epoch: {0}; Loss: {1}".format(i, loss_value))
        pass
    pass
