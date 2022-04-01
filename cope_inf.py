# -*- coding: utf-8 -*-

import numpy as np
from problearner import PMLearner, PALearner
from qlearner_inf import Qlearner_INF
from sklearn.model_selection import KFold


class COPE_INF:
    def __init__(self, dataset,
                 Qlearner_INF, PMLearner, PALearner,
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
        self.target_policy = policy
        if time_difference is None:
            self.time_difference = np.ones(dataset['action'].shape[0])
        else:
            self.time_difference = np.copy(time_difference)

        self.qLearner = Qlearner_INF
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

    def compute_cope_inf(self, s0):
        intercept = self.compute_intercept(s0)
        self.opeuc = np.mean(intercept)
        self.eif_arr = intercept
        return

    def compute_intercept(self, s0):
        num_trajectory = s0.shape[0]
        intercept = np.zeros(num_trajectory)

        ## non-deterministic policy:
        for action_prime in self.unique_action:
            action_prime = action_prime.reshape(1, -1)
            target_pa = np.apply_along_axis(
                self.target_policy, 1, s0, action=action_prime).flatten()
            action_prime_batch = np.repeat(
                action_prime, num_trajectory).flatten().reshape(-1, 1)
            for action_value in self.unique_action:
                action_value = action_value.reshape(1, -1)
                est_pa_value = self.palearner.get_pa_prediction(
                    s0, action_value)
                for mediator_value in self.unique_mediator:
                    mediator_value = mediator_value.reshape(1, -1)
                    est_pm_value = self.pmlearner.get_pm_prediction(
                        s0, action_prime_batch, mediator_value)
                    est_q_value = self.qLearner.get_q_prediction(
                        s0, action_value, mediator_value)
                    intercept_one = est_pm_value * target_pa * est_pa_value * est_q_value
                    intercept += intercept_one
                    # print((mediator_value[0, 0], action_value[0, 0], action_prime[0, 0], intercept_one.mean(), intercept_one.min(), intercept_one.max()))
                    pass
                pass
            pass
        return intercept


def nuisance_estimate_uc_inf(s0, iid_dataset, target_policy, time_difference, gamma,
                             palearner_setting, pmlearner_setting, qlearner_setting):
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
                train_time_difference = time_difference[train_index]
                new_state, new_action, new_mediator, new_reward, new_next_state = iid_dataset[0][test_index], iid_dataset[
                    1][test_index], iid_dataset[2][test_index], iid_dataset[3][test_index], iid_dataset[4][test_index]
                qlearner = Qlearner_INF(iid_dataset_train,
                                        epoch=epoch, verbose=verbose,
                                        model=model, rbf_dim=rbf_dim_value, eps=eps)
                qlearner.fit()
                qlearner.time_difference = time_difference[test_index]
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

    qlearner = Qlearner_INF(iid_dataset, epoch=epoch,
                            verbose=verbose, model=model, rbf_dim=rbf_dim, eps=eps)
    qlearner.fit()

    return qlearner, pmlearner, palearner


def extract_inf_dataset(iid_dataset, num_trajectory, num_time, gamma, time_difference):
    state = np.zeros((num_trajectory, iid_dataset[0].shape[1]))
    action = np.zeros(num_trajectory)
    mediator = np.zeros(num_trajectory)
    for i in range(num_trajectory):
        state[i, ] = iid_dataset[0][i * num_time]
        action[i] = iid_dataset[1][i * num_time]
        mediator[i] = iid_dataset[2][i * num_time]
        pass
    long_reward = np.zeros(num_trajectory)
    for i in range(num_trajectory):
        if time_difference is None:
            discount_reward = 0.0
            for j in range(num_time):
                discount_reward += iid_dataset[3][i*num_time+j] * np.power(gamma, j)
                pass
        else:
            for j in range(num_time):
                t_power = time_difference[i*num_time + j]
                discount_reward += iid_dataset[3][i*num_time+j] * np.power(gamma, t_power)
                pass
        pass
        long_reward[i] = discount_reward
        pass
    inf_iid_dataset = [state, action, mediator, long_reward]
    inf_iid_dataset_dict = {
        'state': state,
        'action': action,
        'mediator': mediator,
        'reward': long_reward
    }
    return inf_iid_dataset, inf_iid_dataset_dict


def cope_inf_run(s0, iid_dataset, target_policy, num_trajectory, num_time,
                 time_difference=None, gamma=0.9,
                 palearner_setting={
                     'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                 pmlearner_setting={'discrete_state': False, 'discrete_action': False,
                                    'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                 qlearner_setting={'epoch': 100,
                                   'verbose': True, 'rbf_dim': None},
                 new_iid_dataset=None, matrix_based_ope=True):

    inf_iid_dataset, inf_iid_dataset_dict = extract_inf_dataset(
        iid_dataset, num_trajectory, num_time, gamma, time_difference)

    qlearner, pmlearner, palearner = nuisance_estimate_uc_inf(
        s0, inf_iid_dataset, target_policy, time_difference, gamma, palearner_setting, pmlearner_setting, qlearner_setting)
    if new_iid_dataset is None:
        dataset = {'s0': s0,
                   'state': iid_dataset[0],
                   'action': iid_dataset[1],
                   'mediator': iid_dataset[2], 'reward': iid_dataset[3],
                   'next_state': iid_dataset[4]}
        opeuc = COPE_INF(dataset, qlearner,
                         pmlearner, palearner, time_difference,
                         gamma, matrix_based_ope, target_policy)
    else:
        opeuc = COPE_INF(inf_iid_dataset_dict, qlearner,
                         pmlearner, palearner, time_difference,
                         gamma, matrix_based_ope, target_policy)
    opeuc.compute_cope_inf(s0)
    return opeuc


def cope_inf_cross_fit(s0, iid_dataset, target_policy, nfold=2, gamma=0.9,
                       palearner_setting={
                           'discrete_state': False, 'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                       pmlearner_setting={'discrete_state': False, 'discrete_action': False,
                                          'rbf_dim': None, 'cv_score': 'accuracy', 'verbose': True},
                       qlearner_setting={'epoch': 100,
                                         'verbose': True, 'rbf_dim': None}, matrix_based_ope=True, split_seed=1):
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

        target_action = np.apply_along_axis(
            target_policy, 1, iid_dataset[0]).flatten()
        target_action_next = np.apply_along_axis(
            target_policy, 1, iid_dataset[4]).flatten()
        # policy_ratio = pmlearner.get_pm_ratio(
        #     iid_dataset[0], target_action, iid_dataset[1], iid_dataset[2])

        remain_iid_index = list(
            set(range(iid_dataset[0].shape[0])).difference(set(part_iid_index)))

        remain_s0 = np.copy(s0)[remain_index, :]
        target_action_s0 = np.apply_along_axis(
            target_policy, 1, remain_s0).flatten()
        remain_iid_dataset = {'s0': remain_s0, 'policy_action_s0': target_action_s0,
                              'state': np.copy(iid_dataset[0])[remain_iid_index, :],
                              'action': np.copy(iid_dataset[1])[remain_iid_index],
                              'policy_action': np.copy(target_action)[remain_iid_index],
                              'policy_action_next': np.copy(target_action_next)[remain_iid_index],
                              'mediator': np.copy(iid_dataset[2])[remain_iid_index],
                              'reward': np.copy(iid_dataset[3])[remain_iid_index],
                              'next_state': np.copy(iid_dataset[4])[remain_iid_index, :]}
        opeuc_obj = cope_inf_run(part_s0, part_iid_data, target_policy, gamma,
                                 palearner_setting, pmlearner_setting, qlearner_setting, remain_iid_dataset, matrix_based_ope)
        opeuc_obj_list.append(opeuc_obj)
        pass
    return opeuc_obj_list
