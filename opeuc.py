# -*- coding: utf-8 -*-

import numpy as np
from problearner import PMLearner, PALearner
from qlearner import Qlearner
from rll import RatioLinearLearner
from sklearn.model_selection import KFold


class OPEUC:
    def __init__(self, dataset,
                 QLearner, RatioLearner,
                 PMLearner, PALearner,
                 time_difference=None, gamma=0.9, 
                 matrix_based_learning=False, 
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
        self.mediator = np.copy(dataset['mediator'])
        self.reward = np.copy(dataset['reward'])
        self.next_state = np.copy(dataset['next_state'])
        self.policy_action = np.copy(dataset['policy_action'])
        self.policy_action_next = np.copy(dataset['policy_action_next'])
        self.s0 = np.copy(dataset['s0'])
        self.policy_action_s0 = np.copy(dataset['policy_action_s0'])
        self.target_policy = policy
        if time_difference is None:
            self.time_difference = np.ones(dataset['action'].shape[0])
        else:
            self.time_difference = np.copy(time_difference)

        self.qLearner = QLearner
        self.ratiolearner = RatioLearner
        self.pmlearner = PMLearner
        self.palearner = PALearner

        self.gamma = gamma

        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.matrix_based_learning = matrix_based_learning
        self.opeuc = None
        self.intercept = None
        self.eif_arr = None
        pass

    def eif_without_intercept(self, data_tuple):
        termI1 = self.compute_termI1(data_tuple)
        termI2 = self.compute_termI2(data_tuple)
        termI3 = self.compute_termI3(data_tuple)
        opeuc = (termI1 + termI2 + termI3) / (1 - self.gamma)
        # print([termI1, termI2, termI3, opeuc])
        try:
            opeuc = opeuc.numpy()[0][0]
        except AttributeError:
            opeuc = opeuc[0]
            # print(opeuc)
        return opeuc
    
    def compute_opeuc(self):
        data_num = self.state.shape[0]
        self.eif_arr = np.array(range(data_num), dtype=float)
        if self.matrix_based_learning:
            intercept_arr = self.compute_intercept_2(self.s0, self.policy_action_s0)
            intercept = np.mean(intercept_arr)
            termI1 = self.compute_termI1_2(self.state, self.action, self.mediator,
                                           self.reward, self.next_state, self.policy_action, self.policy_action_next)
            termI2 = self.compute_termI2_2(self.state, self.action, self.mediator, self.reward, self.next_state, self.policy_action)
            termI3 = self.compute_termI3_2(self.state, self.action, self.mediator, self.reward, self.next_state, self.policy_action)
            # print((termI1[0], termI2[0], termI3[0]))
            # self.eif_arr = (termI1 + termI2 + termI3) / (1 - self.gamma)
            self.eif_arr = termI1 + termI2 + termI3
            self.intercept_arr = np.copy(intercept_arr)
            print(np.array([intercept, np.mean(termI1), np.mean(termI2), np.mean(termI3)]))
            self.eif_arr += intercept
        else:
            intercept = self.compute_intercept(self.s0, self.policy_action_s0)
            for i in range(data_num):
                data_tuple = (self.state[i, :], self.action[i], self.mediator[i],
                            self.reward[i], self.next_state[i], self.policy_action[i])
                self.eif_arr[i] = self.eif_without_intercept(data_tuple) + intercept
                # print(self.eif_arr[i])
                pass
            pass
        
        opeuc = np.mean(self.eif_arr)
        self.opeuc = opeuc
        self.intercept = intercept

        self.cis_arr = self.compute_cis(self.state, self.action, self.mediator, self.reward)
        self.cis = np.mean(self.cis_arr)
        pass

    def compute_termI1(self, data_tuple):
        state = data_tuple[0].reshape(1, -1)
        action = data_tuple[1].reshape(1, -1)
        mediator = data_tuple[2].reshape(1, -1)
        reward = data_tuple[3]
        next_state = data_tuple[4].reshape(1, -1)
        policy_action = data_tuple[5]
        termI1 = reward
        termI1 -= self.qLearner.get_q_prediction(state, action, mediator)
        # print(termI1)
        weight_q_value = 0.0
        # print("start")
        for action_value in self.unique_action:
            for mediator_value in self.unique_mediator:
                action_value = action_value.reshape(1)
                pa_pred = self.palearner.get_pa_prediction(state, action_value)

                action_value = action_value.reshape(1, -1)
                mediator_value = mediator_value.reshape(1, -1)
                pm_pred = self.pmlearner.get_pm_prediction(
                    state, policy_action, mediator_value)
                weight_q_value += self.qLearner.get_q_prediction(
                    next_state, action_value, mediator_value) * pa_pred * pm_pred
                # print([pa_pred, pm_pred, self.qLearner.get_q_prediction(
                #     next_state, action_value, mediator_value), weight_q_value])
                pass
            pass

        termI1 += np.power(self.gamma, self.time_difference) * weight_q_value
        # print(termI1)
        termI1 *= self.ratiolearner.get_r_prediction(state)
        # print(termI1)
        termI1 *= self.pmlearner.get_pm_ratio(state,
                                              policy_action, action, mediator)
        # print(termI1)
        # print("stop")
        return termI1

    def compute_termI1_2(self, state, action, mediator, reward, next_state, policy_action, policy_action_next):
        long_term = np.copy(reward)
        data_point_num = long_term.shape[0]
        time_vary_gamma = np.power(self.gamma, self.time_difference)

        ## non-deterministic policy:
        random_pm_ratio = np.zeros(mediator.shape).flatten()
        for action_prime in self.unique_action:
            action_prime = action_prime.reshape(1, -1)
            target_pa = np.apply_along_axis(self.target_policy, 1, next_state, action=action_prime).flatten()
            action_prime_batch = np.repeat(action_prime, data_point_num).flatten().reshape(-1, 1)
            for action_value in self.unique_action:
                action_value = action_value.reshape(1, -1)
                est_pa_value = self.palearner.get_pa_prediction(next_state, action_value)
                for mediator_value in self.unique_mediator:
                    mediator_value = mediator_value.reshape(1, -1)
                    est_pm_value = self.pmlearner.get_pm_prediction(next_state, action_prime_batch, mediator_value)
                    est_q_value = self.qLearner.get_q_prediction(next_state, action_value, mediator_value)

                    weight_sum_q = time_vary_gamma * est_pm_value * target_pa * est_pa_value * est_q_value
                    long_term += weight_sum_q
                    pass
                pass
            pass
        ## non-deterministic policy:
        for action_prime in self.unique_action:
            action_prime = np.array([action_prime])
            target_pa = np.apply_along_axis(self.target_policy, 1, state, action=action_prime).flatten()
            action_prime = np.repeat(action_prime, mediator.shape[0]).reshape(-1, 1)
            random_pm_ratio += target_pa * self.pmlearner.get_pm_ratio(state, action_prime, action, mediator)

        ## deterministic policy:
        # for action_value in self.unique_action:
        #     action_value = action_value.reshape(1, -1)
        #     for mediator_value in self.unique_mediator:
        #         mediator_value = mediator_value.reshape(1, -1)
        #         # weight_sum_q = self.qLearner.get_q_prediction(next_state, action_value, mediator_value) * self.pmlearner.get_pm_prediction(
        #         #     state, policy_action, mediator_value) * self.palearner.get_pa_prediction(state, action_value)
        #         weight_sum_q = self.qLearner.get_q_prediction(next_state, action_value, mediator_value) * self.pmlearner.get_pm_prediction(
        #             next_state, policy_action_next, mediator_value) * self.palearner.get_pa_prediction(next_state, action_value)
        #         weight_sum_q *= self.gamma
        #         long_term += weight_sum_q
        #         pass
        #     pass
        # random_pm_ratio = self.pmlearner.get_pm_ratio(state, policy_action, action, mediator)

        termI1 = long_term - self.qLearner.get_q_prediction(state, action, mediator)
        termI1 *= self.ratiolearner.get_r_prediction(state)

        
        termI1 *= random_pm_ratio

        termI1 *= 1.0 / (1.0 - time_vary_gamma)
        # termI1 = np.mean(termI1)
        return termI1

    def compute_termI2(self, data_tuple):
        state = data_tuple[0].reshape(1, -1)
        action = data_tuple[1].reshape(1, -1)
        mediator = data_tuple[2].reshape(1, -1)
        policy_action = data_tuple[5].reshape(1, -1)
        termI2 = 0.0
        if action == policy_action:
            # print("start")
            for action_value in self.unique_action:
                action_value = action_value.reshape(1, -1)
                weight_q_value = np.array(0.0)
                for mediator_value in self.unique_mediator:
                    mediator_value = mediator_value.reshape(1, -1)
                    pm_pred = self.pmlearner.get_pm_prediction(
                        state, action, mediator_value)
                    weight_q_one_term = pm_pred
                    weight_q_one_term *= self.qLearner.get_q_prediction(
                        state, action_value, mediator_value)
                    # print(weight_q_one_term)
                    weight_q_value += weight_q_one_term[0]
                    pass
                q_value = self.qLearner.get_q_prediction(
                    state, action_value, mediator)
                termI2_i = (q_value - weight_q_value) * \
                    self.palearner.get_pa_ratio(state, action_value, action)
                termI2 += termI2_i[0]
                # print(termI2)
                pass
            termI2 *= self.ratiolearner.get_r_prediction(state)
            # print("stop")
        return termI2

    def compute_termI2_2(self, state, action, mediator, reward, next_state, policy_action):
        data_point_num = reward.shape
        termI2_complete = np.zeros(data_point_num)
        ## deterministic policy
        # sub_index = np.where(action == policy_action)[0]
        # state = state[sub_index]
        # action = action[sub_index]
        # mediator = mediator[sub_index]
        # reward = reward[sub_index]
        # next_state = next_state[sub_index]

        termI2 = np.zeros(reward.shape)
        for action_value in self.unique_action:
            action_value = action_value.reshape(1, -1)
            q_fix_action = self.qLearner.get_q_prediction(state, action_value, mediator)
            weight_q_sum = np.zeros(reward.shape)
            for mediator_value in self.unique_mediator:
                mediator_value = mediator_value.reshape(1, -1)
                # action_value_batch = np.repeat(action_value, data_point_num).flatten().reshape(-1, 1)
                # pm_est = self.pmlearner.get_pm_prediction(state, action_value_batch, mediator_value)
                pm_est = self.pmlearner.get_pm_prediction(state, action, mediator_value)
                q_est = self.qLearner.get_q_prediction(state, action_value, mediator_value)     
                weight_q_sum += pm_est * q_est
                pass
            q_diff = q_fix_action - weight_q_sum
            q_diff *= self.palearner.get_pa_prediction(state, action_value)
            termI2 += q_diff
            pass
        
        ratio_pred = self.ratiolearner.get_r_prediction(state)
        termI2 *= ratio_pred

        ## deterministic policy
        # termI2_complete[sub_index] = termI2
        
        ## non-deterministic policy
        policy_pa = self.target_policy_pa(self.target_policy, state, action)
        pa_est = self.palearner.get_pa_prediction(state, action)
        pa_ratio = policy_pa / pa_est
        termI2_complete = termI2 * pa_ratio

        time_vary_gamma = np.power(self.gamma, self.time_difference)
        termI2_complete *= 1.0 / (1.0 - time_vary_gamma)

        return termI2_complete

    def compute_termI3(self, data_tuple):
        state = data_tuple[0].reshape(1, -1)
        action = data_tuple[1].reshape(1, -1)
        policy_action = data_tuple[5].reshape(1, -1)
        termI3 = 0
        for mediator_value in self.unique_mediator:
            weight_q_value = 0.0
            mediator_value = mediator_value.reshape(1, -1)
            for action_value in self.unique_action:
                action_value = action_value.reshape(1, -1)
                pa_pred = self.palearner.get_pa_prediction(state, action_value)
                weight_q_value += pa_pred * \
                    self.qLearner.get_q_prediction(
                        state, action_value, mediator_value)
                pass
            q_value = self.qLearner.get_q_prediction(
                state, action, mediator_value)
            termI3 += (q_value - weight_q_value) * \
                self.pmlearner.get_pm_prediction(
                    state, policy_action, mediator_value)
            pass
        termI3 *= self.ratiolearner.get_r_prediction(state)
        return termI3
    
    def compute_termI3_2(self, state, action, mediator, reward, next_state, policy_action):
        termI3 = np.zeros(reward.shape)
        ## non-deterministic policy:
        for action_prime in self.unique_action:
            action_prime = np.array([action_prime])
            policy_pa = np.apply_along_axis(self.target_policy, 1, state, action=action_prime).flatten()
            for mediator_value in self.unique_mediator:
                mediator_value = mediator_value.reshape(1, -1)
                q_fix_mediator = self.qLearner.get_q_prediction(state, action, mediator_value)
                weight_q_sum = np.zeros(reward.shape)
                for action_value in self.unique_action:
                    action_value = action_value.reshape(1, -1)
                    q_est = self.qLearner.get_q_prediction(state, action_value, mediator_value)
                    pa_est = self.palearner.get_pa_prediction(state, action_value)
                    weight_q_sum += q_est * pa_est
                    pass
                q_diff = q_fix_mediator - weight_q_sum
                q_diff *= policy_pa
                action_prime_tmp = np.repeat(action_prime, reward.shape[0]).reshape(-1, 1)
                pm_est = self.pmlearner.get_pm_prediction(state, action_prime_tmp, mediator_value)
                q_diff *= pm_est
                termI3 += q_diff
                pass
            pass

        ## deterministic policy:
        # for mediator_value in self.unique_mediator:
        #     mediator_value = mediator_value.reshape(1, -1)
        #     q_fix_mediator = self.qLearner.get_q_prediction(state, action, mediator_value)
        #     weight_q_sum = np.zeros(reward.shape)
        #     for action_value in self.unique_action:
        #         action_value = action_value.reshape(1, -1)
        #         weight_q_sum += self.qLearner.get_q_prediction(state, action_value, mediator_value) * self.palearner.get_pa_prediction(state, action_value)
        #         pass
        #     q_diff = q_fix_mediator - weight_q_sum
        #     q_diff *= self.pmlearner.get_pm_prediction(state, policy_action, mediator_value)
        #     termI3 += q_diff
        #     pass

        termI3 *= self.ratiolearner.get_r_prediction(state)
        time_vary_gamma = np.power(self.gamma, self.time_difference)
        termI3 *= 1.0 / (1.0 - time_vary_gamma)

        # termI3 = np.mean(termI3)
        return termI3

    def compute_intercept(self, s0, policy_action_s0):
        num_trajectory = s0.shape[0]
        intercept = 0.0
        for i in range(num_trajectory):
            s0_value = s0[i].reshape(1, -1)
            for action_value in self.unique_action:
                for mediator_value in self.unique_mediator:
                    mediator_value = mediator_value.reshape(1, -1)
                    action_value = action_value.reshape(1, -1)
                    est_q_value = self.qLearner.get_q_prediction(
                        s0_value, action_value, mediator_value)
                    est_pm_value = self.pmlearner.get_pm_prediction(
                        s0_value, policy_action_s0[i].reshape(1, -1), mediator_value)
                    est_pa_value = self.palearner.get_pa_prediction(
                        s0_value, action_value)
                    intercept += est_q_value * est_pm_value * est_pa_value
        intercept /= (1.0 * num_trajectory)
        return intercept
    
    def compute_intercept_2(self, s0, policy_action_s0):
        num_trajectory = s0.shape[0]
        intercept = np.zeros(num_trajectory)

        ## non-deterministic policy:
        for action_prime in self.unique_action:
            action_prime = action_prime.reshape(1, -1)
            target_pa = np.apply_along_axis(self.target_policy, 1, s0, action=action_prime).flatten()
            action_prime_batch = np.repeat(action_prime, num_trajectory).flatten().reshape(-1, 1)
            for action_value in self.unique_action:
                action_value = action_value.reshape(1, -1)
                est_pa_value = self.palearner.get_pa_prediction(s0, action_value)
                for mediator_value in self.unique_mediator:
                    mediator_value = mediator_value.reshape(1, -1)
                    est_pm_value = self.pmlearner.get_pm_prediction(s0, action_prime_batch, mediator_value)
                    est_q_value = self.qLearner.get_q_prediction(s0, action_value, mediator_value)
                    intercept_one = est_pm_value * target_pa * est_pa_value * est_q_value
                    intercept += intercept_one
                    # print((mediator_value[0, 0], action_value[0, 0], action_prime[0, 0], intercept_one.mean(), intercept_one.min(), intercept_one.max()))
                    pass
                pass
            pass

        ## deterministic policy: 
        # for action_value in self.unique_action:
        #     for mediator_value in self.unique_mediator:
        #         mediator_value = mediator_value.reshape(1, -1)
        #         action_value = action_value.reshape(1, -1)
        #         est_q_value = self.qLearner.get_q_prediction(
        #             s0, action_value, mediator_value)
        #         est_pm_value = self.pmlearner.get_pm_prediction(
        #             s0, policy_action_s0.reshape(1, -1), mediator_value)
        #         est_pa_value = self.palearner.get_pa_prediction(
        #             s0, action_value)
        #         intercept += est_q_value * est_pm_value * est_pa_value

        # intercept = np.mean(intercept)
        return intercept

    def compute_cis(self, state, action, mediator, reward):
        '''
        Confounded important sampling method.
        '''
        data_point_num = reward.shape
        weight_pm = np.zeros(data_point_num)
        for action_prime in self.unique_action:
            action_prime = action_prime.reshape(1, -1)
            target_pa = np.apply_along_axis(self.target_policy, 1, state, action=action_prime).flatten()
            action_prime_batch = np.repeat(action_prime, data_point_num).flatten().reshape(-1, 1)
            est_pm_value = self.pmlearner.get_pm_prediction(state, action_prime_batch, mediator)
            weight_pm += target_pa * est_pm_value
            pass
        pm_est = self.pmlearner.get_pm_prediction(state, action, mediator)
        pm_ratio = weight_pm / pm_est
        is_est = pm_ratio * reward * self.ratiolearner.get_r_prediction(state)
        time_vary_gamma = np.power(self.gamma, self.time_difference)
        is_est = is_est / (1 - time_vary_gamma)

        return is_est

    def get_opeuc(self):
        return self.opeuc

    def target_policy_pa(self, target_policy, state, action):
        num = action.shape[0]
        target_pa = list(range(num))
        for i in range(num):
            target_pa[i] = target_policy(state[i], action[i])
            pass
        target_pa = np.array(target_pa).flatten()
        return target_pa


def nuisance_estimate_uc(s0, iid_dataset, target_policy, time_difference, gamma,
                         palearner_setting, pmlearner_setting, qlearner_setting, ratiolearner_setting):
    ## Train conditional probability of action given state
    discrete_state = palearner_setting['discrete_state']
    rbf_dim = palearner_setting['rbf_dim']
    cv_score = palearner_setting['cv_score']
    verbose = palearner_setting['verbose']
    palearner = PALearner(iid_dataset, discrete_state,
                          rbf_dim, cv_score, verbose)
    palearner.train()

    ## Train conditional probability of mediator given state and action
    discrete_state = pmlearner_setting['discrete_state']
    discrete_action = pmlearner_setting['discrete_action']
    rbf_dim = pmlearner_setting['rbf_dim']
    cv_score = pmlearner_setting['cv_score']
    verbose = palearner_setting['verbose']
    pmlearner = PMLearner(iid_dataset, discrete_state,
                          discrete_action, rbf_dim, cv_score, verbose)
    pmlearner.train()

    ## Train Q-estimator
    epoch = qlearner_setting['epoch']
    rbf_dim = qlearner_setting['rbf_dim']
    verbose = qlearner_setting['verbose']
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
                iid_dataset_train = [iid_dataset[0][train_index], iid_dataset[1][train_index],
                                     iid_dataset[2][train_index], iid_dataset[3][train_index],
                                     iid_dataset[4][train_index]]
                new_state, new_action, new_mediator, new_reward, new_next_state = iid_dataset[0][test_index], iid_dataset[
                    1][test_index], iid_dataset[2][test_index], iid_dataset[3][test_index], iid_dataset[4][test_index]
                qlearner = Qlearner(iid_dataset_train, target_policy, pmlearner,
                                    palearner, time_difference=time_difference, 
                                    gamma=gamma, epoch=epoch, verbose=verbose,
                                    model=model, rbf_dim=rbf_dim_value, eps=eps)
                qlearner.fit()
                rmse_arr[index] += qlearner.goodness_of_fit(
                    target_policy, new_state, new_action, new_mediator, new_reward, new_next_state)
                pass
            pass
        rbf_dim = optional_rbf_dim[np.argmin(rmse_arr)]
        if verbose:
            print("Optimal RBF feature of Q-estimator:", rbf_dim)
    elif type(rbf_dim) is list:
        rbf_dim = rbf_dim[0]
    else:
        pass

    qlearner = Qlearner(iid_dataset, target_policy, pmlearner,
                        palearner, time_difference=time_difference, 
                        gamma=gamma, epoch=epoch,
                        verbose=verbose, model=model, rbf_dim=rbf_dim, eps=eps)
    qlearner.fit()

    ## Train Ratio estimator:
    ratio_rbf_dim = ratiolearner_setting['rbf_ndims']
    rlearner_type = ratiolearner_setting['mode']
    prespecific_rbf_dim_candidate = type(ratio_rbf_dim) is list and len(ratio_rbf_dim) > 1
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
                                     'action': iid_dataset[1][train_index], 'mediator': iid_dataset[2][train_index]}
                    rationearner = RatioLinearLearner(dataset_train, target_policy, pmlearner, 
                                                      time_difference=time_difference, gamma=gamma, 
                                                      ndim=rbf_dim_value)
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
                   "next_state": iid_dataset[4], 'action': iid_dataset[1], 'mediator': iid_dataset[2]}
        rationearner = RatioLinearLearner(dataset, target_policy, pmlearner, 
                                          time_difference=time_difference, gamma=gamma, 
                                          ndim=ratio_rbf_dim)
        rationearner.fit()
        # rll_prediction = rationearner.get_ratio_prediction(iid_dataset[0])
        # print("RLL prediction: ", (np.quantile(
        #     rll_prediction, q=np.array([0.0, 0.25, 0.5, 0.75, 1.0])), np.mean(rll_prediction)))
        pass

    return qlearner, rationearner, pmlearner, palearner

def opeuc_run(s0, iid_dataset, target_policy, time_difference=None, gamma=0.9,
              palearner_setting={'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
              pmlearner_setting={'discrete_state': False, 'discrete_action': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
              qlearner_setting={'epoch': 100, 'verbose': True, 'rbf_dim': None},
              ratiolearner_setting={'mode': 'linear', 'rbf_ndims': None,
                                    'batch_size': 32, 'epoch': 100, 'lr': 0.01, 'verbose': True},
              new_iid_dataset=None, matrix_based_ope=True):
    qlearner, rationearner, pmlearner, palearner = nuisance_estimate_uc(
        s0, iid_dataset, target_policy, time_difference, gamma, palearner_setting, pmlearner_setting, qlearner_setting, ratiolearner_setting)
    if new_iid_dataset is None:
        np.random.seed(1)
        target_action_s0 = np.apply_along_axis(target_policy, 1, s0).flatten()
        np.random.seed(1)
        target_action = np.apply_along_axis(target_policy, 1, iid_dataset[0]).flatten()
        np.random.seed(1)
        target_action_next = np.apply_along_axis(target_policy, 1, iid_dataset[4]).flatten()
        # policy_ratio = pmlearner.get_pm_ratio(
        #     iid_dataset[0], target_action, iid_dataset[1], iid_dataset[2])
        # dataset = {'s0': s0, 'policy_action_s0': target_action_s0,
        #            'state': iid_dataset[0],
        #            'action': iid_dataset[1], 'policy_action': target_action,
        #            'mediator': iid_dataset[2], 'reward': iid_dataset[3],
        #            'next_state': iid_dataset[4], 'policy_ratio': policy_ratio}
        dataset = {'s0': s0, 'policy_action_s0': target_action_s0,
                   'state': iid_dataset[0],
                   'action': iid_dataset[1], 'policy_action': target_action,
                   'policy_action_next': target_action_next,
                   'mediator': iid_dataset[2], 'reward': iid_dataset[3],
                   'next_state': iid_dataset[4]}
        opeuc = OPEUC(dataset, qlearner, rationearner,
                      pmlearner, palearner, time_difference, gamma, matrix_based_ope, target_policy)
    else:
        opeuc = OPEUC(new_iid_dataset, qlearner, rationearner,
                      pmlearner, palearner, time_difference, gamma, matrix_based_ope, target_policy)
    opeuc.compute_opeuc()
    return opeuc

def opeuc_cross_fit(s0, iid_dataset, target_policy, nfold=2, gamma=0.9,
                    palearner_setting={
                        'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                    pmlearner_setting={'discrete_state': False, 'discrete_action': False,
                                       'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                    qlearner_setting={'epoch': 100,
                                      'verbose': True, 'rbf_dim': None},
                    ratiolearner_setting={'mode': 'linear', 'rbf_ndims': None,
                                          'batch_size': 32, 'epoch': 100, 'lr': 0.01, 'verbose': True}, matrix_based_ope=True, split_seed=1):
    trajectory_num = s0.shape[0]
    time_num = int(iid_dataset[0].shape[0] / trajectory_num)
    np.random.seed(split_seed)
    random_index = np.arange(trajectory_num)
    np.random.shuffle(random_index)

    opeuc_obj_list = []
    part_num = int(trajectory_num/2)
    for i in range(nfold):
        if i + 1 == nfold:
            part_index = range(i*part_num, trajectory_num)
        else:
            part_index = range(i*part_num, (i+1)*part_num)
        part_index = random_index[part_index]

        part_s0 = np.copy(s0)[part_index, :]
        remain_index = list(
            set(range(trajectory_num)).difference(set(part_index)))

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
        # qlearner, rationearner, pmlearner, palearner = nuisance_estimate_uc(
            # part_s0, part_iid_data, target_policy, gamma, palearner_setting, pmlearner_setting, qlearner_setting, ratiolearner_setting)

        target_action = np.apply_along_axis(target_policy, 1, iid_dataset[0]).flatten()
        target_action_next = np.apply_along_axis(target_policy, 1, iid_dataset[4]).flatten()
        # policy_ratio = pmlearner.get_pm_ratio(
        #     iid_dataset[0], target_action, iid_dataset[1], iid_dataset[2])

        remain_iid_index = list(set(range(iid_dataset[0].shape[0])).difference(set(part_iid_index)))

        remain_s0 = np.copy(s0)[remain_index, :]
        target_action_s0 = np.apply_along_axis(target_policy, 1, remain_s0).flatten()
        remain_iid_dataset = {'s0': remain_s0, 'policy_action_s0': target_action_s0,
                              'state': np.copy(iid_dataset[0])[remain_iid_index, :],
                              'action': np.copy(iid_dataset[1])[remain_iid_index], 
                              'policy_action': np.copy(target_action)[remain_iid_index],
                              'policy_action_next': np.copy(target_action_next)[remain_iid_index], 
                              'mediator': np.copy(iid_dataset[2])[remain_iid_index], 
                              'reward': np.copy(iid_dataset[3])[remain_iid_index],
                              'next_state': np.copy(iid_dataset[4])[remain_iid_index, :]}
        opeuc_obj = opeuc_run(part_s0, part_iid_data, target_policy, gamma,
                              palearner_setting, pmlearner_setting, qlearner_setting, ratiolearner_setting, remain_iid_dataset, matrix_based_ope)
        opeuc_obj_list.append(opeuc_obj)
        pass
    return opeuc_obj_list

