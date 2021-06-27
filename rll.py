# -*- coding: utf-8 -*-

import numpy as np
from problearner import PMLearner, PALearner
from numpy.linalg import inv
from sklearn.kernel_approximation import RBFSampler

class RatioLinearLearner:
    '''
    Input
    --------
    cplearner is an object of PMLearner or PALearner. 
    It gives estimators for conditional probability of behaviour policy: 
    P(action|state) (if input a PALearner), P(mediator|action, state) (if input a PMLearner).

    Examples
    --------
    num_iid_point = 1000
    num_trajectory = 20
    num_T = int(num_iid_point / num_trajectory)
    dim_state = 3 # no NaN
    dim_state = 10
    s0 = np.random.normal(size=(num_trajectory, dim_state))
    for i in range(num_trajectory):
        next_state = np.random.normal(size=(num_T-1)*dim_state).reshape(num_T-1, dim_state)
        state = np.vstack([s0[i, :], next_state[range(num_T - 2), :]])
        if i == 0:
            state_all = state
            next_state_all = next_state
        else:
            state_all = np.vstack([state_all, state])
            next_state_all = np.vstack([next_state_all, next_state])
    action = np.random.choice([75, 80, 85, 90, 100], 
                            num_iid_point-num_trajectory)
    mediator = np.random.choice([75, 80, 85, 90, 100], 
                                num_iid_point-num_trajectory)
    reward = np.random.normal(size=num_iid_point-num_trajectory)
    dataset = {'s0': s0, 'state': state_all, 
            'action': action, 'mediator': mediator, 
            'reward': reward, 
            "next_state": next_state_all}
    nn_setting = {'epoch': 5}

    def toy_policy(state):
        action = np.random.choice([75, 80, 85, 90, 100], size=1)
        return action

    iid_dataset = [state_all, action, mediator, reward, next_state_all]
    pmlearner = PMLearner(iid_dataset)
    pmlearner.train()
    rlearner = RatioLinearLearner(dataset, toy_policy, pmlearner, ndim = 80)
    rlearner.train()

    for i in range(num_trajectory):
        state = np.vstack([s0[i, :], next_state[range(num_T - 2), :]])
        if i == 0:
            state_all_test = state
        else:
            state_all_test = np.vstack([state_all_test, state])
    ratio_pred = rlearner.get_ratio_prediction(state_all_test)
    np.mean(ratio_pred)  ## close to 1

    ## close to 1 with different RBF features number
    dim_list =  np.linspace(5, 100, endpoint=True, num=20, dtype=int)
    res_list = []
    for ndim in dim_list:
        rlearner = RatioLinearLearner(dataset, toy_policy, pmlearner, ndim = ndim)
        rlearner.train()
        res_list.append(np.mean(rlearner.get_ratio_prediction(state_all_test)))
        pass
    res = np.vstack([dim_list, res_list])
    res.transpose()
    '''
    def __init__(self, dataset, policy, cplearner, time_difference=None, gamma=0.9, ndim=100, l2penalty=1.0, use_mediator=True):
        self.use_mediator = use_mediator

        self.state = np.copy(dataset['state'])
        self.action = np.copy(dataset['action']).reshape(-1, 1)
        self.unique_action = np.unique(dataset['action'])
        if use_mediator:
            self.mediator = np.copy(dataset['mediator']).reshape(-1, 1)
        self.next_state = np.copy(dataset['next_state'])
        self.s0 = np.copy(dataset['s0'])
        if time_difference is None:
            self.time_difference = np.ones(self.action.shape[0])
        else:
            self.time_difference = np.copy(time_difference)

        self.policy = policy
        self.cplearner = cplearner

        self.gamma = gamma
        self.l2penalty = l2penalty
        self.beta = None
        self.rbf_feature = RBFSampler(random_state=1, n_components=ndim)
        self.rbf_feature.fit(np.vstack((self.state, self.s0)))
        pass

    def feature_engineering(self, feature):
        feature_new = self.rbf_feature.transform(feature)
        feature_new = np.hstack([np.repeat(1, feature_new.shape[0]).reshape(-1, 1), feature_new])
        return feature_new

    def target_policy_pa(self, target_policy, state, action):
        num = action.shape[0]
        target_pa = list(range(num))
        for i in range(num):
            target_pa[i] = target_policy(state[i], action[i])
            pass
        target_pa = np.array(target_pa).flatten()
        return target_pa

    def fit(self):
        psi = self.feature_engineering(self.state)
        psi_next = self.feature_engineering(self.next_state)
        if self.use_mediator:
            ## non-deterministic policy:
            ratio = np.zeros(self.action.shape)
            for action_value in self.unique_action:
                action_value = np.array([action_value])
                target_pa = np.apply_along_axis(self.policy, 1, self.state, action=action_value).flatten()
                policy_action_tmp = np.repeat(action_value, self.action.shape[0]).reshape(-1, 1)
                pm_ratio = self.cplearner.get_pm_ratio(self.state, policy_action_tmp, self.action, self.mediator)
                # print(pm_ratio.shape)
                # print(policy_action_tmp.flatten().shape)
                # print(ratio.shape)
                ratio += (pm_ratio * policy_action_tmp.flatten()).reshape(-1, 1)
            ratio = ratio.flatten()
            ## deterministic policy:
            # self.policy_action = np.array([self.policy(state_value)[0] for state_value in self.state])
            # pm_ratio = self.cplearner.get_pm_ratio(self.state, self.policy_action, self.action, self.mediator)
        else:
            estimate_pa = self.cplearner.get_pa_prediction(self.state, self.action)
            target_pa = self.target_policy_pa(self.policy, self.state, self.action)
            pa_ratio = target_pa / estimate_pa
            ratio = pa_ratio
        # print(np.mean(ratio)) # close to 1 if behaviour and target are the same
        psi_minus_psi_next = self.rbf_difference(psi, psi_next, ratio)
        design_matrix = np.zeros((psi.shape[1], psi.shape[1]))
        for i in range(self.state.shape[0]):
            design_matrix += np.matmul(psi[i].reshape(-1, 1), psi_minus_psi_next[i].reshape(1, -1))
        # design_matrix = np.matmul(psi.transpose(), psi_minus_psi_next)
        design_matrix /= self.state.shape[0]
        # print(design_matrix)
        penalty_matrix = np.diagflat(np.repeat(self.l2penalty, design_matrix.shape[0]))
        # if psi.shape[0] <= psi.shape[1]:
        #     penalty_matrix = np.diagflat(np.repeat(self.l2penalty, design_matrix.shape[0]))
        # else:
        #     penalty_matrix = np.zeros(design_matrix.shape)
        penalize_design_matrix = design_matrix + penalty_matrix
        inv_design_matrix = inv(penalize_design_matrix)

        # psi_s0 = self.feature_engineering(self.s0)
        # mean_psi_s0 = (1 - self.gamma) * np.mean(psi_s0, axis=0)
        # print(mean_psi_s0)
        mean_psi_s0 = self.ratio_expectation_s0(np.copy(self.s0))

        beta = np.matmul(inv_design_matrix, mean_psi_s0.reshape(-1, 1))
        self.beta = beta
        pass
    
    def rbf_difference(self, psi, psi_next, ratio):
        # psi_next = self.gamma * (psi_next.transpose() * ratio).transpose()
        psi_next = np.multiply((psi_next.transpose() * ratio).transpose(),
                               np.power(self.gamma, self.time_difference)[:, np.newaxis])
        psi_minus_psi_next = psi - psi_next
        return psi_minus_psi_next

    def get_ratio_prediction(self, state, truncate=20, normalize=True):
        '''
        Input:
        state: a numpy.array
        Output:
        A 1D numpy array. The probability ratio in certain states.
        '''
        if np.ndim(state) == 0 or np.ndim(state) == 1:
            x_state = np.reshape(state, (1, -1))
        else:
            x_state = np.copy(state)
        psi = self.feature_engineering(x_state)
        ratio = np.matmul(psi, self.beta).flatten()
        ratio_min = 1 / truncate
        ratio_max = truncate
        ratio = np.clip(ratio, a_min=ratio_min, a_max=ratio_max)
        if state.shape[0] > 1:
            if normalize:
                ratio /= np.mean(ratio)
        return ratio
    
    def ratio_expectation_s0(self, s0):
        psi_s0 = self.feature_engineering(s0)
        mean_psi_s0 = (1 - self.gamma) * np.mean(psi_s0, axis=0)
        return mean_psi_s0

    def get_r_prediction(self, state, truncate=20, normalize=True):
        return self.get_ratio_prediction(state, truncate, normalize)

    def goodness_of_fit(self, target_policy, new_s0, new_state, new_action, new_mediator, new_reward, new_next_state):
        np.random.seed(1)
        new_policy_action = np.apply_along_axis(target_policy, 1, new_state)
        if self.use_mediator:
            pm_ratio = self.cplearner.get_pm_ratio(new_state, new_policy_action, new_action, new_mediator)
            new_cp_ratio = pm_ratio
        else:
            estimate_pa = self.cplearner.get_pa_prediction(new_state, new_action)
            target_pa = self.target_policy_pa(self.policy, new_state, new_action)
            pa_ratio = target_pa / estimate_pa
            new_cp_ratio = pa_ratio
        
        phi_new_state = self.feature_engineering(new_state)
        phi_new_next_state = self.feature_engineering(new_next_state)
        new_ratio = self.get_ratio_prediction(new_state)
        psi_minus_psi_next = self.rbf_difference(phi_new_state, phi_new_next_state, new_ratio)

        ratio_weighted_psi_minus_psi_next = np.multiply(
            psi_minus_psi_next, new_cp_ratio[:, np.newaxis])
        # print(psi_minus_psi_next[0:2])
        # print(new_cp_ratio[0:2])
        # print(ratio_weighted_psi_minus_psi_next[0:2])
        mean_ratio_weighted_psi_minus_psi_next = np.mean(ratio_weighted_psi_minus_psi_next, axis=0)

        mean_new_psi_s0 = self.ratio_expectation_s0(new_s0)

        rmse = np.sqrt(np.mean(np.square(mean_ratio_weighted_psi_minus_psi_next - mean_new_psi_s0)))
        return rmse



        
        
        

