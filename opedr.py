# -*- coding: utf-8 -*-

import numpy as np
from problearner import PMLearner, PALearner
from qlearner import Qlearner
from rll import RatioLinearLearner
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


class OPEDR:
    def __init__(self, dataset,
                 QLearner, RatioLearner, PALearner,
                 gamma, target_policy, 
                 time_difference=None,
                 matrix_based_learning=False, 
                 time_point_num=None):
        '''


        Parameters
        ----------
        dataset : A Dict
            A list with 6 elements. 
            They are: state, action, mediator, 
            reward, next state, action under policy to be evaluted.
        policy : TYPE
            DESCRIPTION.
        QLearner : TYPE
            A Q-learning model.
        RationLearner : TYPE
            A deep learning model for learning policy ratio.
        PMLearner : TYPE
            DESCRIPTION.
        PALearner : TYPE
            DESCRIPTION.
        gamma : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.state = np.copy(dataset['state'])
        self.action = np.copy(dataset['action'])
        self.unique_action = np.unique(self.action)
        self.mediator = np.copy(dataset['mediator'])
        self.reward = np.copy(dataset['reward'])
        self.next_state = np.copy(dataset['next_state'])
        self.policy_action = np.copy(dataset['policy_action'])
        self.s0 = np.copy(dataset['s0'])
        self.policy_action_s0 = np.copy(dataset['policy_action_s0'])

        self.target_policy = target_policy
        self.qLearner = QLearner
        self.ratiolearner = RatioLearner
        self.palearner = PALearner

        if time_difference is None:
            self.time_difference = np.ones(dataset['action'].shape[0])
        else:
            self.time_difference = np.copy(time_difference)
        self.matrix_based_learning = matrix_based_learning
        self.gamma = gamma
        self.time_point_num = time_point_num

        self.opedr = None
        self.opedr2 = None
        self.intercept = None
        self.eif_arr = None
        self.weight_reward = None
        pass

    def eif_without_intercept(self, data_tuple):
        term_debias = self.compute_termI1(data_tuple)
        opedr = term_debias / (1 - self.gamma)
        # print(term_debias)
        try:
            opedr = opedr.numpy()[0][0]
        except AttributeError:
            opedr = opedr
        return opedr

    def compute_opedr(self):
        data_num = self.state.shape[0]
        self.eif_arr = np.array(range(data_num), dtype=float)

        if self.matrix_based_learning:
            intercept = self.compute_intercept_2(self.s0, self.policy_action_s0)
            # termI1 = self.compute_termI1_2(self.state, self.action, self.reward, self.next_state, self.policy_action)
            # self.eif_arr = termI1 / (1.0 - self.gamma)
            termI1 = self.compute_termI1_3(self.state, self.action, self.reward, self.next_state, self.policy_action)
            self.eif_arr = termI1
            self.eif_arr += intercept
            # weight_reward = self.compute_term_weight_reward(
            #     self.state, self.action, self.reward, self.next_state, self.policy_action)
            # self.weight_reward = weight_reward / (1.0 - self.gamma)
            weight_reward = self.compute_term_weight_reward2(self.state, self.action, self.reward, self.next_state, self.policy_action)
            self.weight_reward = weight_reward
        else:
            intercept = self.compute_intercept(self.s0, self.policy_action_s0)
            for i in range(data_num):
                data_tuple = (self.state[i, :], self.action[i], self.mediator[i],
                            self.reward[i], self.next_state[i], self.policy_action[i])
                self.eif_arr[i] = self.eif_without_intercept(
                    data_tuple) + intercept
                # print(self.eif_arr[i])
        opedr = np.mean(self.eif_arr)
        self.opedr = opedr
        if intercept.ndim == 2:
            intercept = intercept[0][0]
        elif intercept.ndim == 1:
            intercept = intercept[0]
        else:
            pass
        self.intercept = intercept

    def compute_termI1(self, data_tuple):
        state = data_tuple[0].reshape(1, -1)
        action = data_tuple[1].reshape(1, -1)
        next_state = data_tuple[4].reshape(1, -1)
        policy_next_state_action = self.target_policy(next_state)
        policy_action = data_tuple[5].reshape(1, -1)
        reward = data_tuple[3]
        termI2 = 0.0
        if action == policy_action:
            pred_reward = self.gamma * \
                self.qLearner.get_q_prediction(
                    next_state, policy_next_state_action)
            pred_reward += reward
            reward_bias = pred_reward - \
                self.qLearner.get_q_prediction(state, action)
            ratio = self.ratiolearner.get_r_prediction(state)
            # ratio *= self.target_policy(state, policy_action) / self.palearner.get_pa_prediction(state, action)
            ratio *= 1.0 / self.palearner.get_pa_prediction(state, action)
            reward_bias *= ratio
            termI2 = reward_bias
        return termI2
    
    def compute_termI1_2(self, state, action, reward, next_state, policy_action):
        termI1_complete = np.zeros(reward.shape)

        sub_index = np.where(policy_action == action)[0]
        state = state[sub_index]
        action = action[sub_index]
        reward = reward[sub_index]
        next_state = next_state[sub_index]
        
        np.random.seed(1)
        next_policy_action = np.apply_along_axis(self.target_policy, 1, next_state)

        ratio_state = self.ratiolearner.get_r_prediction(state)
        prob_action = self.palearner.get_pa_prediction(state, action)
        q_fix_action = self.qLearner.get_q_prediction(state, action)
        q_long_time = reward
        q_est = self.qLearner.get_q_prediction(next_state, next_policy_action)
        time_vary_gamma = np.power(self.gamma, self.time_difference)
        q_long_time += time_vary_gamma * q_est
        q_diff = q_long_time - q_fix_action
        term_I1 = q_diff * ratio_state / prob_action

        termI1_complete[sub_index] = term_I1
        return termI1_complete
    
    def compute_term_weight_reward(self, state, action, reward, next_state, policy_action):
        term_weight_reward_complete = np.zeros(reward.shape)

        sub_index = np.where(policy_action == action)[0]
        state = state[sub_index]
        action = action[sub_index]
        reward = reward[sub_index]

        ratio_state = self.ratiolearner.get_r_prediction(state)
        prob_action = self.palearner.get_pa_prediction(state, action)
        term_weight_reward = reward * ratio_state / prob_action

        term_weight_reward_complete[sub_index] = term_weight_reward
        return np.mean(term_weight_reward_complete)
    
    def target_policy_pa(self, target_policy, state, action):
        num = action.shape[0]
        target_pa = list(range(num))
        for i in range(num):
            target_pa[i] = target_policy(state[i], action[i])
            pass
        target_pa = np.array(target_pa).flatten()
        return target_pa

    def compute_termI1_3(self, state, action, reward, next_state, policy_action):
        termI1_complete = np.zeros(reward.shape)
        target_pa = self.target_policy_pa(self.target_policy, state, action)
        # next_policy_action = np.apply_along_axis(self.target_policy, 1, next_state)

        prob_action = self.palearner.get_pa_prediction(state, action)
        ratio_state = self.ratiolearner.get_r_prediction(state)
        q_fix_action = self.qLearner.get_q_prediction(state, action)
        q_long_time = np.copy(reward)
        time_vary_gamma = np.power(self.gamma, self.time_difference)
        for action_value in self.unique_action:
            action_value = np.array([action_value])
            target_next_state_pa = np.apply_along_axis(self.target_policy, 1, next_state, action=action_value).flatten()
            action_value = np.repeat(action_value, next_state.shape[0])
            q_est = self.qLearner.get_q_prediction(next_state, action_value) * target_next_state_pa
            q_long_time += time_vary_gamma * q_est
        q_diff = q_long_time - q_fix_action
        termI1_complete = q_diff * ratio_state * target_pa / prob_action

        termI1_complete *= 1.0 / (1.0 - self.gamma)
        return termI1_complete

    def compute_term_weight_reward2(self, state, action, reward, next_state, policy_action):
        ratio_state = self.ratiolearner.get_r_prediction(state)
        target_pa = self.target_policy_pa(self.target_policy, state, action)
        prob_action = self.palearner.get_pa_prediction(state, action)
        term_weight_reward = reward * ratio_state * target_pa / prob_action
        time_vary_gamma = np.power(self.gamma, self.time_difference)
        term_weight_reward = term_weight_reward / (1.0 - time_vary_gamma)
        self.is_arr = np.copy(term_weight_reward)
        term_weight_reward = np.mean(term_weight_reward)
        return term_weight_reward

    def compute_termI1_5(self, state, action, reward, next_state, policy_action, ratio_product):
        termI1_complete = np.zeros(reward.shape)

        sub_index = np.where(policy_action == action)[0]
        state = state[sub_index]
        action = action[sub_index]
        reward = reward[sub_index]
        next_state = next_state[sub_index]
        ratio_product = ratio_product[sub_index]
        np.random.seed(1)
        next_policy_action = np.apply_along_axis(self.target_policy, 1, next_state)

        prob_action = self.palearner.get_pa_prediction(state, action)
        q_fix_action = self.qLearner.get_q_prediction(state, action)
        q_long_time = reward
        q_long_time += self.gamma * self.qLearner.get_q_prediction(next_state, next_policy_action)
        q_diff = q_long_time - q_fix_action
        term_I1 = q_diff * ratio_product / prob_action

        termI1_complete[sub_index] = term_I1
        return termI1_complete

    def compute_intercept(self, s0, policy_action_s0):
        num_trajectory = s0.shape[0]
        intercept = 0.0
        for i in range(num_trajectory):
            s0_value = s0[i].reshape(1, -1)
            policy_action_value = policy_action_s0[i].reshape(1, -1)
            intercept += self.qLearner.get_q_prediction(
                s0_value, policy_action_value)
        intercept /= (1.0 * num_trajectory)
        return intercept
    
    def compute_intercept_2(self, s0, policy_action_s0):
        ## deterministic policy
        intercept = np.zeros(s0.shape[0]).flatten()
        for action_value in self.unique_action:
            action_value = np.array([action_value])
            target_pa = np.apply_along_axis(self.target_policy, 1, s0, action=action_value).flatten()
            action_value = np.repeat(action_value, s0.shape[0]).flatten()
            q_pred = self.qLearner.get_q_prediction(s0, action_value)
            # print(intercept.shape)
            # print(q_pred.shape)
            # print(target_pa.shape)
            intercept += q_pred * target_pa

        ## non-deterministic policy 
        # intercept = self.qLearner.get_q_prediction(s0, policy_action_s0)
        self.intercept_arr = np.copy(intercept)
        intercept = np.mean(intercept)
        return intercept

    def compute_ratio_product(self):
        ratio_product = []
        i = 0
        for time_length in self.time_point_num:
            ratio_product_one = []
            for t in range(time_length):
                state_one = self.state[i]
                action_one = self.action[i]
                evaluate_prob = self.target_policy(state_one, action_one)
                state_one = state_one.reshape(1, -1)
                action_one = action_one.reshape(-1, 1)
                behavior_prob = self.palearner.get_pa_prediction(state_one, action_one)
                ratio_product_one.append(evaluate_prob / behavior_prob)
                i = i + 1
                pass
            ratio_product_one = np.cumprod(ratio_product_one)
            ratio_product.extend(ratio_product_one)
            pass
        ratio_product = np.array(ratio_product)
        ratio_product_returned = np.copy(ratio_product)
        return ratio_product_returned

    def compute_opedr2(self):
        if self.time_point_num is None:
            print("Jiang and Li (2016) Estimator is not allow.")
            raise ValueError

        if self.matrix_based_learning:
            ratio_product = self.compute_ratio_product()
            intercept = self.compute_intercept_2(self.s0, self.policy_action_s0)
            termI1 = self.compute_termI1_5(self.state, self.action, self.reward, self.next_state, self.policy_action, ratio_product)
            self.eif_arr = termI1 / (1.0 - self.gamma)
            self.eif_arr += intercept
        else:
            print("Jiang and Li (2016) Estimator is not support non-matrix based learning.")
            raise ValueError

        opedr2 = np.mean(self.eif_arr)
        self.intercept = intercept
        self.opedr2 = opedr2

    def get_opedr(self):
        return self.opedr


def nuisance_estimate_dr(s0, iid_dataset, target_policy, time_difference, gamma, palearner_setting, qlearner_setting, ratiolearner_setting):
    ## Train conditional probability of action given state
    discrete_state = palearner_setting['discrete_state']
    rbf_dim = palearner_setting['rbf_dim']
    cv_score = palearner_setting['cv_score']
    verbose = palearner_setting['verbose']
    palearner = PALearner(iid_dataset, discrete_state,
                          rbf_dim, cv_score, verbose)
    palearner.train()

    ## Learing Q estimator
    epoch = qlearner_setting['epoch']
    verbose = qlearner_setting['verbose']
    rbf_dim = qlearner_setting['rbf_dim']
    model = qlearner_setting['model']
    eps = qlearner_setting['eps']
    prespecific_rbf_dim_candidate = type(rbf_dim) is list and len(rbf_dim) > 1
    if rbf_dim is None or prespecific_rbf_dim_candidate:
        nfold = 5
        kf = KFold(n_splits=nfold, shuffle=True, random_state=1)
        if rbf_dim is None:
            ## select model for fitted q iteration:
            if model == "linear":
                dim_start = s0.shape[1]
                dim_end = dim_start * 20
                cv_size = 30
            elif model == "forest":
                dim_start = 3
                dim_end = 30
                cv_size = 10
            else:
                pass
            if dim_end - dim_start < cv_size:
                cv_size = dim_end - dim_start
            optional_rbf_dim = np.linspace(
                dim_start, dim_end, num=cv_size, dtype=int)
        else:
            optional_rbf_dim = np.array(rbf_dim).flatten()

        rmse_arr = np.zeros(optional_rbf_dim.shape)
        for index, rbf_dim_value in enumerate(optional_rbf_dim):
            for train_index, test_index in kf.split(iid_dataset[0]):
                # print("TRAIN:", train_index, "TEST:", test_index)
                iid_dataset_train = [iid_dataset[0][train_index], iid_dataset[1][train_index], iid_dataset[2][train_index], iid_dataset[3][train_index],
                                  iid_dataset[4][train_index]]
                qlearner = Qlearner(iid_dataset, target_policy,
                                    None, palearner, time_difference=time_difference, 
                                    gamma=gamma, epoch=epoch, verbose=verbose, model=model,
                                    rbf_dim=rbf_dim_value, use_mediator=False, eps=eps)
                qlearner.fit()

                new_state, new_action, new_mediator, new_reward, new_next_state = iid_dataset[0][test_index], iid_dataset[
                    1][test_index], iid_dataset[2][test_index], iid_dataset[3][test_index], iid_dataset[4][test_index]
                rmse_arr[index] += qlearner.goodness_of_fit(target_policy, new_state, new_action, new_mediator, new_reward, new_next_state)
                pass
            pass
        rbf_dim  = optional_rbf_dim[np.argmin(rmse_arr)]
        if verbose:
            print("Optimal RBF feature of Q-estimator:", rbf_dim)
            pass
        pass
    elif type(rbf_dim) is list:
        rbf_dim = rbf_dim[0]
        pass
    else:
        pass

    qlearner = Qlearner(iid_dataset, target_policy, None, palearner, time_difference=time_difference, 
                        gamma=gamma, epoch=epoch, verbose=verbose, model=model,
                        rbf_dim=rbf_dim, use_mediator=False, eps=eps)
    qlearner.fit()

    ## Train Ratio estimator:
    ratio_rbf_dim = ratiolearner_setting['rbf_ndims']
    rlearner_type = ratiolearner_setting['mode']
    verbose = ratiolearner_setting['verbose']
    prespecific_rbf_dim_candidate = type(
        ratio_rbf_dim) is list and len(ratio_rbf_dim) > 1
    rlearner_type = 'linear'
    if rlearner_type == 'linear':
        if ratio_rbf_dim is None or prespecific_rbf_dim_candidate:
            nfold = 5
            dim_start = s0.shape[1]
            dim_end = dim_start * 50
            kf_s0 = KFold(n_splits=nfold, shuffle=True, random_state=1)
            kf = KFold(n_splits=nfold, shuffle=True, random_state=1)
            if ratio_rbf_dim is None:
                optional_rbf_dim = np.linspace(
                    dim_start, dim_end, num=50, dtype=int)
            else:
                optional_rbf_dim = np.array(ratio_rbf_dim).flatten()
            rmse_arr = np.zeros(optional_rbf_dim.shape)

            for index, rbf_dim_value in enumerate(optional_rbf_dim):
                for index_state, index_s0 in zip(kf.split(iid_dataset[0]), kf.split(s0)):
                    train_index = index_state[0]
                    test_index = index_state[1]
                    s0_train_index = index_s0[0]
                    s0_test_index = index_s0[1]
                    dataset_train = {'s0': s0[s0_train_index], 'state': iid_dataset[0][train_index], "next_state": iid_dataset[4][train_index],
                                     'action': iid_dataset[1][train_index]}
                    rationearner = RatioLinearLearner(
                        dataset_train, target_policy, palearner, time_difference=time_difference, 
                        ndim=rbf_dim_value, use_mediator=False)
                    rationearner.fit()

                    new_s0, new_state, new_action, new_mediator, new_reward, new_next_state = s0[s0_test_index], iid_dataset[0][test_index], iid_dataset[
                        1][test_index], iid_dataset[2][test_index], iid_dataset[3][test_index], iid_dataset[4][test_index]
                    rmse_arr[index] += rationearner.goodness_of_fit(
                        target_policy, new_s0, new_state, new_action, new_mediator, new_reward, new_next_state)
                    pass
                pass

            ratio_rbf_dim = optional_rbf_dim[np.argmin(rmse_arr)]
            if verbose: 
                print("Optimal RBF feature of Ratio-estimator:", ratio_rbf_dim)
        elif type(ratio_rbf_dim) is list:
            ratio_rbf_dim = ratio_rbf_dim[0]
        else:
            pass

        dataset = {'s0': s0, 'state': iid_dataset[0],
                    "next_state": iid_dataset[4], 'action': iid_dataset[1]}
        rationearner = RatioLinearLearner(
            dataset, target_policy, palearner, time_difference=time_difference, 
            ndim=ratio_rbf_dim, use_mediator=False)
        rationearner.fit()
        pass

    return qlearner, rationearner, palearner

def opedr_run(s0, iid_dataset, target_policy, time_difference=None, gamma=0.9,
              palearner_setting={'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
              qlearner_setting={'epoch': 100, 'trace': True, 'rbf_dim': 5},
              ratiolearner_setting={'mode': 'linear', 'rbf_ndims': 5,
                                    'batch_size': 32, 'epoch': 100, 'lr': 0.01, 'trace': True},
              new_iid_dataset=None, matrix_based_ope=True, time_point_num=None, opedr2=False):
    '''
    Inputs: 
    new_iid_dataset: a helpful parameter used for cross fitting
    '''
    qlearner, rationearner, palearner = nuisance_estimate_dr(
        s0, iid_dataset, target_policy, time_difference, gamma, palearner_setting, qlearner_setting, ratiolearner_setting)
    if new_iid_dataset is None:
        np.random.seed(1)
        target_action_s0 = np.apply_along_axis(target_policy, 1, s0).flatten()
        np.random.seed(1)
        target_action = np.apply_along_axis(target_policy, 1, iid_dataset[0]).flatten()
        dataset = {'s0': s0, 'policy_action_s0': target_action_s0,
                   'state': iid_dataset[0],
                   'action': iid_dataset[1], 'policy_action': target_action,
                   'mediator': iid_dataset[2], 'reward': iid_dataset[3],
                   'next_state': iid_dataset[4]}
        opedr = OPEDR(dataset, qlearner, rationearner,
                      palearner, gamma, target_policy, time_difference, matrix_based_ope, time_point_num)
    else:
        opedr = OPEDR(new_iid_dataset, qlearner, rationearner,
                      palearner, gamma, target_policy, time_difference, matrix_based_ope, time_point_num)
    if opedr2:
        opedr.compute_opedr2()
    else:
        opedr.compute_opedr()
    return opedr

def opedr_cross_fit(s0, iid_dataset, target_policy, nfold=2, gamma=0.9,
                    palearner_setting={'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                    qlearner_setting={'epoch': 100,'verbose': True, 'rbf_dim': None},
                    ratiolearner_setting={'mode': 'linear', 'rbf_ndims': 1, 'batch_size': 32, 'epoch': 100, 'lr': 0.01, 'verbose': True}, 
                    matrix_based_ope=True, split_seed=1, time_point_num=None, opedr2=False):
    trajectory_num = s0.shape[0]
    time_num = int(iid_dataset[0].shape[0] / trajectory_num)
    np.random.seed(split_seed)
    random_index = np.arange(trajectory_num)
    np.random.shuffle(random_index)

    opedr_obj_list = []
    part_num = int(trajectory_num/2)
    for i in range(nfold):
        if i + 1 == nfold:
            part_index = range(i*part_num, trajectory_num)
        else:
            part_index = range(i*part_num, (i+1)*part_num)
        part_index = random_index[part_index]

        part_s0 = np.copy(s0)[part_index, :]
        remain_index = list(set(range(trajectory_num)).difference(set(part_index)))

        part_iid_index = np.array([])
        for index in part_index:
            part_iid_index = np.append(
                part_iid_index, index * time_num + np.arange(time_num))
            pass
        part_iid_index = part_iid_index.astype(int)
        part_iid_index = part_iid_index.tolist()

        part_iid_data = [np.copy(iid_dataset[0])[part_iid_index, :],
                         np.copy(iid_dataset[1])[part_iid_index],
                         np.copy(iid_dataset[2])[part_iid_index],
                         np.copy(iid_dataset[3])[part_iid_index],
                         np.copy(iid_dataset[4])[part_iid_index, :]]

        np.random.seed(1)
        target_action = np.apply_along_axis(target_policy, 1, iid_dataset[0]).flatten()

        remain_iid_index = list(
            set(range(iid_dataset[0].shape[0])).difference(set(part_iid_index)))

        remain_s0 = np.copy(s0)[remain_index, :]
        np.random.seed(1)
        target_action_s0 = np.apply_along_axis(target_policy, 1, remain_s0).flatten()
        remain_iid_dataset = {'s0': remain_s0, 'policy_action_s0': target_action_s0,
                              'state': np.copy(iid_dataset[0])[remain_iid_index, :],
                              'action': np.copy(iid_dataset[1])[remain_iid_index], 'policy_action': np.copy(target_action)[remain_iid_index],
                              'mediator': np.copy(iid_dataset[2])[remain_iid_index], 'reward': np.copy(iid_dataset[3])[remain_iid_index],
                              'next_state': np.copy(iid_dataset[4])[remain_iid_index, :]}
        opedr_obj = opedr_run(part_s0, part_iid_data, target_policy, gamma,
                              palearner_setting, qlearner_setting, ratiolearner_setting, remain_iid_dataset, 
                              matrix_based_ope, time_point_num, opedr2)
        opedr_obj_list.append(opedr_obj)
        pass
    return opedr_obj_list
