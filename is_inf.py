# -*- coding: utf-8 -*-

from matplotlib.pyplot import axis
import numpy as np
from problearner import PALearner as CondProbLearner

class IS_INF:
    def __init__(self, dataset,
                 cond_prob_learner_list, 
                 time_difference, 
                 gamma=0.9, 
                 policy=None):
        '''
        
        Parameters
        ----------
        dataset : A Dict
            A list with 6 elements. 
            They are: state, action, mediator, 
            reward, next state, action under policy to be evaluted.
        policy : TYPE
            DESCRIPTION.
        CondProbLearner : TYPE
            DESCRIPTION.
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.history_dataset = dataset
        self.reward = dataset[3]
        self.target_policy = policy
        self.cp_estimator_list = cond_prob_learner_list
        self.num_time = dataset[1].shape[1]
        if time_difference is None:
            self.time_difference = np.ones(self.num_time)
        else:
            self.time_difference = np.copy(time_difference)

        self.gamma = gamma
        self.unique_action = np.unique(dataset[1])
        self.target_policy = policy
        pass
    
    def compute_is_inf(self):  
        ratio_list = []
        for t in range(self.num_time):
            train_x, train_y = get_time_t_train_data(self.history_dataset, t)
            pm_pred = self.cp_estimator_list[t].model_prediction(train_x, train_y)
            weight_pm_pred = np.zeros(pm_pred.shape)

            last_state = get_time_t_state(self.history_dataset, t)
            for a in self.unique_action:
                train_x[:, -1] = a
                pm_pred_action = self.cp_estimator_list[t].model_prediction(train_x, train_y)
                action_prob = self.target_policy(last_state, a)
                weight_pm_pred += action_prob * pm_pred_action
                pass
            ratio_list.append((weight_pm_pred / pm_pred).reshape(-1, 1))
            pass
        ratio_mat = np.concatenate(ratio_list, axis=1)
        ratio_mat = np.cumprod(ratio_mat, axis=1)
        num_trajectory = ratio_mat.shape[0]
        gamma_weight = np.ones((num_trajectory, 1)) * np.power(self.gamma, range(self.num_time)).reshape(1, -1)
        est_mat = gamma_weight * ratio_mat * self.reward

        # result:
        self.eif_arr = np.sum(est_mat, axis=1)
        self.opeuc = self.eif_arr.mean()

def data_processing(trajectory_dataset):
    '''
    Output: a list containing:
    1. 3d state matrix (trajectory by the first axis, time by the second axis, dimension by the third axis)
    2. 2d action matrix (trajectory by row, time by column)
    3. 2d mediator matrix
    4. 2d reward matrix
    '''
    num_trajectory = len(trajectory_dataset)
    state_list = list()
    action_list = list()
    mediator_list = list()
    reward_list = list()
    num_time = trajectory_dataset[0][0].shape[0]
    dim_state = trajectory_dataset[0][0].shape[1]

    for i in range(num_trajectory):
        state_i = trajectory_dataset[i][0]
        state_i = np.reshape(state_i, (1, num_time, dim_state))
        state_list.append(state_i)
        action_list.append(trajectory_dataset[i][1].reshape(1, -1))
        mediator_list.append(trajectory_dataset[i][2].reshape(1, -1))
        reward_list.append(trajectory_dataset[i][3].reshape(1, -1))
        pass
    state_mat = np.concatenate(state_list, axis=0)
    mediator_mat = np.concatenate(mediator_list, axis=0)
    action_mat = np.concatenate(action_list, axis=0)
    reward_mat = np.concatenate(reward_list, axis=0)

    trajectory_dataset = [state_mat, action_mat, mediator_mat, reward_mat]
    return trajectory_dataset

def get_time_t_state(processed_trajectory, time):
    state_t = processed_trajectory[0][:, time, :]
    return state_t

def get_time_t_train_data(processed_trajectory, time):
    state_list = []
    num_trajectory = processed_trajectory[0].shape[0]
    for i in range(num_trajectory):
        state_i = processed_trajectory[0][i, :, :]
        state_vector = state_i[range(time+1), :].reshape(1, -1)
        state_list.append(state_vector)
        pass
    state_mat = np.concatenate(state_list, axis=0)
    action_mat = processed_trajectory[1][:, range(time+1)]
    mediator_mat = processed_trajectory[2][:, range(time+1)]
    train_y = mediator_mat[:, -1]
    mediator_mat = np.delete(mediator_mat, -1, axis=1)
    train_x = np.concatenate([state_mat, mediator_mat, action_mat], axis=1)
    return train_x, train_y

def nuisance_estimate(processed_trajectory, cond_prob_learner_setting):
    ## Train conditional probability of mediator given state and action
    discrete_state = cond_prob_learner_setting['discrete_state']
    rbf_dim = cond_prob_learner_setting['rbf_dim']
    cv_score = cond_prob_learner_setting['cv_score']
    verbose = cond_prob_learner_setting['verbose']

    num_time = processed_trajectory[1].shape[1]
    cp_estimator_list = []
    for i in range(num_time):
        train_x, train_y = get_time_t_train_data(processed_trajectory, i)
        train_data = [train_x, train_y]
        cond_prob_learner = CondProbLearner(train_data, discrete_state, rbf_dim, cv_score, verbose)
        cond_prob_learner.train()
        cp_estimator_list.append(cond_prob_learner)
        pass

    return cp_estimator_list

def is_inf_run(trajectory_dataset, target_policy, time_difference=None, gamma=0.9,
              cond_prob_learner_setting={'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
              new_iid_dataset=None):
    processed_trajectory = data_processing(trajectory_dataset)
    cp_estimator_list = nuisance_estimate(
        processed_trajectory, cond_prob_learner_setting)
    if new_iid_dataset is None:
        is_inf = IS_INF(processed_trajectory, cp_estimator_list, time_difference, gamma, target_policy)
    else:
        pass
    is_inf.compute_is_inf()
    return is_inf
