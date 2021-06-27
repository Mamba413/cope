# -*- coding: utf-8 -*-

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from problearner import PMLearner, PALearner
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

class Qlearner:
    def __init__(self, data, target_policy, PMLearner, PALearner, time_difference=None, gamma=0.9,
                 epoch=1000, verbose=False, model='forest', rbf_dim=5, use_mediator=True, eps=1e-2):
        '''

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        Examples
        -------
        >>> import pickle
        >>> import os
        >>> os.chdir("../raw")
        >>> with open('iid_dataset.pkl', 'rb') as f:
        >>>     iid_dataset = pickle.load(f)

        >>> palearner = PALearner(iid_dataset)
        >>> palearner.train()
        >>> pmlearner = PMLearner(iid_dataset)
        >>> pmlearner.train()
        >>> random_policy = np.copy(iid_dataset[1])
        >>> np.random.shuffle(random_policy)

        >>> qlearner = Qlearner(iid_dataset, random_policy, pmlearner, palearner, epoch=5, verbose=True)
        >>> qlearner.train()
        '''
        self.data = data
        self.action = np.copy(data[1])
        self.mediator = np.copy(data[2])
        self.action = np.reshape(self.action, (-1, 1))
        self.mediator = np.reshape(self.mediator, (-1, 1))
        
        self.reward = np.copy(data[3])
        self.pmlearner = PMLearner
        self.palearner = PALearner
        if time_difference is None:
            self.time_difference = np.ones(self.reward.shape[0])
        else:
            self.time_difference = np.copy(time_difference)
        self.gamma = gamma

        self.target_policy = target_policy
        np.random.seed(1)
        policy_action = np.apply_along_axis(target_policy, 1, data[4])
        self.policy_action = policy_action

        self.unique_action = np.unique(data[1])
        if self.unique_action.ndim == 1:
            tmp_unique_action = self.unique_action.reshape(-1, 1)
        self.ohe_action_feature = OneHotEncoder(
            drop='first', sparse=False).fit(tmp_unique_action)
        self.action_dummy_size = np.size(
            self.ohe_action_feature.categories_[0]) - 1

        self.use_mediator = use_mediator
        if use_mediator:
            self.unique_mediator = np.unique(data[2])
            if self.unique_mediator.ndim == 1:
                tmp_unique_mediator = self.unique_mediator.reshape(-1, 1)
            self.ohe_mediator_feature = OneHotEncoder(
                drop='first', sparse=False).fit(tmp_unique_mediator)
            self.mediator_dummy_size = np.size(
                self.ohe_mediator_feature.categories_[0]) - 1
        else:
            pass

        self.model = model
        hyperparameter = rbf_dim
        if self.model == "linear":
            self.rbf_feature = RBFSampler(random_state=1, n_components=hyperparameter)
            X_action = self.ohe_action_feature.fit_transform(self.action)
            if X_action.ndim == 1:
                X_action = X_action.reshape(-1, self.action_dummy_size)
            if self.use_mediator:
                X_mediator = self.ohe_mediator_feature.transform(self.mediator)
                if X_mediator.ndim == 1:
                    X_mediator = X_mediator.reshape(-1, self.mediator_dummy_size)
                X_am = np.hstack((X_action, X_mediator))
            else:
                X_am = X_action

            ## transform and concat
            # X_state = self.rbf_feature.fit_transform(self.data[0])
            # X = np.hstack((X_state, X_am))

            ## concat and transform
            X_state = np.copy(self.data[0])
            X = np.hstack((X_state, X_am))
            if self.model == "linear":
                self.train_X = self.rbf_feature.fit_transform(X)

            # self.q_model = LinearRegression(random_state=1)
            self.q_model = Ridge(solver='lsqr', alpha=1e-5, random_state=1)
        elif model == "forest":
            # print(hyperparameter)
            # self.q_model = RandomForestRegressor(
            # max_depth=6, random_state=1, min_samples_leaf=hyperparameter)
            self.q_model = RandomForestRegressor(random_state=1, min_samples_leaf=hyperparameter)
        else:
            pass

        self.epoch = epoch
        self.score_array = np.zeros(epoch)
        self.bias_array = np.zeros(epoch)
        self.rmse_array = np.zeros(epoch)
        self.rmedianse_array = np.zeros(epoch)
        self.q_model_list = []
        self.verbose = verbose
        self.eps = eps
        pass

    def pesudo_response(self, reward, next_state, policy_action):
        q_next_state = np.zeros(shape=reward.shape).flatten()
        if self.use_mediator:
            ## Implementation 1:
            # long_term_reward = self.pmlearner.get_pm_prediction(
            #     next_state, policy_action, mediator)
            # for action_value in self.unique_action:
            #     # value = np.ones(shape=reward.shape)
            #     action_value = np.array([action_value])
            #     value = self.palearner.get_pa_prediction(
            #         next_state, action_value)
            #     q_value = self.get_q_prediction(
            #         next_state, action_value.reshape(1, -1), mediator).flatten()
            #     value *= q_value
            #     # print("Q-value: ", q_value)
            #     scale_value += value
            # long_term_reward *= scale_value
            # long_term_reward *= self.gamma
            # long_term_reward += reward.flatten()

            ## Implementation V function (version 2):
            ## non-deterministic policy:
            for action_star in self.unique_action:
                action_star = np.array([action_star])
                target_pa = np.apply_along_axis(self.target_policy, 1, next_state, action=action_star).flatten()
                one_policy_action = np.repeat(action_star, next_state.shape[0]).reshape(-1, 1)
                for action_value in self.unique_action:
                    action_value = np.array([action_value])
                    pa_pred = self.palearner.get_pa_prediction(next_state, action_value)
                    for mediator_value in self.unique_mediator:
                        mediator_value = np.array([mediator_value])
                        pm_pred = self.pmlearner.get_pm_prediction(next_state, one_policy_action, mediator_value)
                        q_next_state_prob = pm_pred * pa_pred * target_pa
                        q_next_state_prob *= self.get_q_prediction(next_state, action_value, mediator_value)
                        q_next_state += q_next_state_prob
                        pass
                    pass
                pass

            ## deterministic policy:
            # for mediator_value in self.unique_mediator:
            #     mediator_value = np.array([mediator_value])
            #     pm_pred = self.pmlearner.get_pm_prediction(next_state, policy_action, mediator_value)
            #     for action_value in self.unique_action:
            #         action_value = np.array([action_value])
            #         pa_pred = self.palearner.get_pa_prediction(next_state, action_value)
            #         q_next_state_prob = pm_pred * pa_pred
            #         q_next_state_prob *= self.get_q_prediction(next_state, action_value, mediator_value)
            #         q_next_state += q_next_state_prob
            #         pass
            #     pass
        else:
            for action_value in self.unique_action:
                action_value = np.array([action_value])
                pa_pred = np.apply_along_axis(self.target_policy, 1, next_state, action=action_value).flatten()
                # pa_pred = self.palearner.get_pa_prediction(next_state, action_value)
                q_value = self.get_q_prediction(next_state, action_value.reshape(1, -1)).flatten()
                q_next_state += pa_pred * q_value
                pass
            pass

        long_term_reward = reward.flatten()
        # long_term_reward += self.gamma * q_next_state
        long_term_reward += np.power(self.gamma, self.time_difference) * q_next_state
        # # print("Pesudo response: ", long_term_reward)
        return long_term_reward

    def one_batch_fit(self):
        pesudo_y = self.pesudo_response(np.copy(self.data[3]), np.copy(self.data[4]), np.copy(self.policy_action))

        self.q_model = clone(self.q_model)
        self.q_model.fit(self.train_X, pesudo_y)
        self.q_model_list.append(self.q_model)
        self.score_array[self.iteration_time] = self.q_model.score(self.train_X, pesudo_y)
        error = pesudo_y - self.q_model.predict(self.train_X)
        self.bias_array[self.iteration_time] = np.mean(error)
        self.rmse_array[self.iteration_time] = np.sqrt(np.mean(np.square(error)))
        self.rmedianse_array[self.iteration_time] = np.sqrt(np.median(np.square(error)))
        # if self.verbose:
        #     print(self.q_model.score(X, pesudo_y))
        pass

    def fit(self):
        # self.preprocess()
        for i in range(self.epoch):
            self.iteration_time = i
            self.one_batch_fit()
            if i >= 1:
                score_difference = self.score_array[i] - self.score_array[i - 1]
                rmse_difference = self.rmse_array[i] - self.rmse_array[i - 1]
                if self.model == "linear":
                    coef_diff1 = self.q_model_list[i].coef_ - self.q_model_list[i - 1].coef_
                    coef_diff1 = np.linalg.norm(coef_diff1, ord=1)
                    coef_diff2 = np.abs(self.q_model_list[i].intercept_ - self.q_model_list[i - 1].intercept_)
                    coef_diff = coef_diff1 + coef_diff2
                    coef_norm = np.linalg.norm(self.q_model_list[i].coef_, ord=1)
                    coef_norm += np.abs(self.q_model_list[i].intercept_)
                    relative_coef_diff = coef_diff / coef_norm
                    # print((coef_diff, coef_norm, relative_coef_diff))
                    pass
                # if score_difference < 1e-3 and self.iteration_time >= 5:
                if relative_coef_diff < self.eps or coef_diff < 1e-4:
                    if rmse_difference > 0:
                        self.q_model = self.q_model_list[i - 1]
                    else:
                        self.q_model = self.q_model_list[i]
                    break
                pass
            pass
        index = range(self.iteration_time+1)
        self.score_array = self.score_array[index]
        self.bias_array = self.bias_array[index]
        self.rmse_array = self.rmse_array[index]
        self.rmedianse_array = self.rmedianse_array[index]

    def get_q_prediction(self, state, action, mediator=None):
        if self.iteration_time == 0:
            np.random.seed(1)
            if self.use_mediator:
                pred = np.random.normal(size=mediator.shape)
            else:
                pred = np.random.normal(size=(state.shape[0], 1))
            # pred = np.copy(self.reward)
            pred = pred.flatten()
        else:
            if action.shape[0] != state.shape[0]:
                action = np.repeat(action, state.shape[0], axis=0)
            if action.ndim == 1:
                # action = action.reshape(-1, self.action_dummy_size)
                action = action.reshape(-1, 1)
            x_action = self.ohe_action_feature.transform(action)

            if self.use_mediator:
                if mediator.shape[0] != state.shape[0]:
                    mediator = np.repeat(mediator, state.shape[0], axis=0)
                if mediator.ndim == 1:
                    # mediator = mediator.reshape(-1, self.mediator_dummy_size)
                    mediator = mediator.reshape(-1, 1)
                x_mediator = self.ohe_mediator_feature.transform(mediator)
                x_am = np.hstack((x_action, x_mediator))
            else:
                x_am = x_action

            ## tranform and concat
            # x_state = self.rbf_feature.fit_transform(state)
            # x = np.hstack((x_state, x_am))

            ## concat and tranform
            x_state = np.copy(state)
            x = np.hstack((x_state, x_am))
            if self.model == "linear":
                x = self.rbf_feature.transform(x)

            pred = self.q_model.predict(x)

        return pred

    def goodness_of_fit(self, target_policy, new_state, new_action, new_mediator, new_reward, new_next_state):
        np.random.seed(1)
        new_policy_action = np.apply_along_axis(target_policy, 1, new_next_state)
        y = self.pesudo_response(new_reward, new_next_state, new_policy_action)
        y_pred = self.get_q_prediction(new_state, new_action, new_mediator)
        rmse = np.sqrt(np.mean(np.square(y - y_pred)))
        return rmse

