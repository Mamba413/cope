# coding: utf-8

import numpy as np
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import make_scorer

class PMLearner():
    def __init__(self, data, discrete_state=False, discrete_action=False, rbf_dim=None, cv_score='accuracy', verbose=False):
        '''

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        discrete_action : BOOL
            if True, action will be treated as a discrete variable and converts to a series of dummy variables.
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
        
        >>> pmlearner = PMLearner(iid_dataset)
        >>> pmlearner.train()

        >>> print(pmlearner.model_prediction(iid_dataset[0][:3, :], iid_dataset[1][:3], iid_dataset[2][:3]))
        >>> print(pmlearner.get_pm_prediction(iid_dataset[0][:3, :], iid_dataset[1][:3], iid_dataset[2][:3]))
        >>> print(pmlearner.get_pm_ratio(iid_dataset[0][:3, :], iid_dataset[1][:3], iid_dataset[2][:3], np.array([90, 80, 75])))

        '''
        self.data = data
        self.model = None
        self.discrete_state = discrete_state
        self.discrete_action = discrete_action
        self.ohe_state = OneHotEncoder(drop="first")
        self.ohe_action = OneHotEncoder(drop="first")
        self.poly = PolynomialFeatures(interaction_only=True)
        self.rbf_feature = None
        self.rbf_dim = rbf_dim
        self.cv_score = cv_score
        self.fit_feature_engineering()
        self.verbose = verbose

    def fit_feature_engineering(self):
        state = np.copy(self.data[0])
        action = np.copy(self.data[1])
        self.ohe_state.fit(state)
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        self.ohe_action.fit(action)

    def train(self):
        state = np.copy(self.data[0])
        action = np.copy(self.data[1])
        mediator = np.copy(self.data[2])

        if self.discrete_state:
            state = self.ohe_state.transform(state)
        feature_state = state

        if self.discrete_action:
            action = self.ohe_action.transform(action)
        if action.ndim == 1:
            action = action.reshape(-1, 1)
        # feature_action = self.poly.fit_transform(action)
        feature_action = action

        X = np.hstack([feature_state, feature_action])
        y = mediator

        condition1 = self.rbf_dim is None
        condition2 = type(self.rbf_dim) is list
        if condition1 or condition2:
            pipeline = Pipeline([
                ('rbf', RBFSampler(random_state=1)),
                ('logistic', LogisticRegression(penalty='l2', C=20.0, random_state=0)),
            ])
            if condition1:
                para_start = X.shape[1]
                para_end = para_start * 20
                parameters = {
                    'rbf__n_components': np.linspace(para_start, para_end, num=30, dtype=int),
                }
            elif condition2:
                parameters = {
                    'rbf__n_components': np.array(self.rbf_dim),
                }
            else:
                pass

            if self.cv_score == 'accuracy':
                scorer = make_scorer(accuracy_score)
            elif self.cv_score == 'mcc':
                scorer = make_scorer(matthews_corrcoef)
            else:
                pass
            grid_search = GridSearchCV(
                pipeline, parameters, n_jobs=-1, verbose=0, refit=False, scoring=scorer)
            grid_search.fit(X, y)
            self.rbf_dim = grid_search.best_params_['rbf__n_components']
            if self.verbose:
                print("Optimal RBF feature of PMLearner:", self.rbf_dim)

        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.rbf_dim)
        # X = self.rbf_feature.fit_transform(X)
        self.model = LogisticRegression(penalty='l2', C=1.0, random_state=1).fit(X, y)
        self.data = None

    def model_prediction(self, state, action, mediator):
        if self.discrete_state:
            state = self.ohe_state.transform(state)
        if state.ndim == 1:
            state = state.reshape(1, state.size)
        # state = self.poly.fit_transform(state)

        if state.ndim == 1:
            action = action.reshape(1, 1)
        else:
            action = action.reshape((action.size, 1))
        if self.discrete_action:
            action = self.ohe_action.transform(action)
        # action = self.poly.fit_transform(action)

        x = np.hstack([state, action])
        # x = self.rbf_feature.transform(x)
        y_prob = self.model.predict_proba(x)
        if state.ndim == 1:
            prob_index = np.where(self.model.classes_ == mediator)[0].tolist()[0]
            pred = y_prob[0][prob_index]
        else:
            if mediator.shape[0] != 1:
                pred = []
                prob_index = [np.where(self.model.classes_ == one_mediator)[0].tolist()[0] for one_mediator in mediator]
                for i, index in enumerate(prob_index):
                    pred.append(y_prob[i, index])
            else:
                prob_index = np.where(self.model.classes_ == mediator[0])[0].tolist()[0]
                pred = y_prob[:, prob_index]

        pred = np.array(pred)
        return pred

    def get_pm_prediction(self, state, action, mediator):
        pm_prediction = self.model_prediction(state, action, mediator)
        return pm_prediction

    def get_pm_ratio(self, state, action1, action2, mediator):
        pm_prediction1 = self.model_prediction(state, action1, mediator)
        pm_prediction2 = self.model_prediction(state, action2, mediator)
        return pm_prediction1 / pm_prediction2

    def get_m_prediction(self, state, action):
        if self.discrete_state:
            state = self.ohe_state.transform(state)
        if state.ndim == 1:
            state = state.reshape(1, state.size)
        # state = self.poly.fit_transform(state)

        if state.ndim == 1:
            action = action.reshape(1, 1)
        else:
            action = action.reshape((action.size, 1))
        if self.discrete_action:
            action = self.ohe_action.transform(action)
        # action = self.poly.fit_transform(action)

        x = np.hstack([state, action])
        x = self.rbf_feature.transform(x)
        mediator_pred = self.model.predict(x)
        return mediator_pred


class PALearner():
    def __init__(self, data, discrete_state=False, rbf_dim=None, cv_score='accuracy', verbose=False):
        '''
        Examples
        -------
        >>> import pickle
        >>> import os
        >>> os.chdir("../raw")
        >>> with open('iid_dataset.pkl', 'rb') as f:
        >>>     iid_dataset = pickle.load(f)
        
        >>> palearner = PALearner(iid_dataset)
        >>> palearner.train()

        >>> print(palearner.model_prediction(iid_dataset[0][:3, :], [75, 80, 90]))
        >>> print(palearner.get_pa_prediction(iid_dataset[0][:3, :], [75, 80, 90]))
        >>> print(palearner.get_pa_ratio(iid_dataset[0][:3, :], [75, 80, 90], [90, 80, 75]))
        '''
        self.data = data
        self.model = None
        self.discrete_state = discrete_state
        self.ohe_state = OneHotEncoder(drop="first")
        self.poly = PolynomialFeatures(interaction_only=True)
        self.rbf_dim = rbf_dim
        self.rbf_feature = None
        self.cv_score = cv_score
        self.fit_feature_engineering()
        self.verbose = verbose

    def fit_feature_engineering(self):
        state = np.copy(self.data[0])
        self.ohe_state.fit(state)

    def train(self):
        state = np.copy(self.data[0])
        action = np.copy(self.data[1])
        if self.discrete_state:
            state = self.ohe_state.transform(state)
        X = state
        y = action

        condition1 = self.rbf_dim is None
        condition2 = type(self.rbf_dim) is list
        if condition1 or condition2:
            pipeline = Pipeline([
                ('rbf', RBFSampler(random_state=1)),
                ('logistic', LogisticRegression(penalty='l2', C=20.0, random_state=0)),
            ])
            if condition1:
                para_start = X.shape[1]
                para_end = para_start * 20
                parameters = {
                    'rbf__n_components': np.linspace(para_start, para_end, num=30, dtype=int),
                }
            elif condition2:
                parameters = {
                    'rbf__n_components': np.array(self.rbf_dim),
                }
            else:
                pass

            if self.cv_score == 'accuracy':
                scorer = make_scorer(accuracy_score)
            elif self.cv_score == 'mcc':
                scorer = make_scorer(matthews_corrcoef)
            else:
                pass
            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0, refit=False, scoring=scorer)
            grid_search.fit(X, y)
            self.rbf_dim = grid_search.best_params_['rbf__n_components']
            if self.verbose:
                print("Optimal RBF feature of PALearner:", self.rbf_dim)

        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.rbf_dim)

        X = self.rbf_feature.fit_transform(X)
        # X = self.poly.fit_transform(X)

        self.model = LogisticRegression(penalty='l2', C=20.0, random_state=1).fit(X, y)
        self.data = None

    def model_prediction(self, state, action):
        if self.discrete_state:
            state = self.ohe_state.transform(state)
        if state.ndim == 1:
            state = state.reshape(1, state.size)
        # x = self.poly.fit_transform(state)
        x = self.rbf_feature.transform(state)
        y_prob = self.model.predict_proba(x)
        if state.ndim == 1:
            prob_index = np.where(self.model.classes_ == action)[0].tolist()[0]
            pred = y_prob[0][prob_index]
        else:
            if action.shape[0] != 1:
                pred = []
                prob_index = [np.where(self.model.classes_ == one_action)[0].tolist()[0] for one_action in action]
                for i, index in enumerate(prob_index):
                    pred.append(y_prob[i, index])
            else:
                prob_index = np.where(self.model.classes_ == action[0])[0].tolist()[0]
                pred = y_prob[:, prob_index]
                pass
            pass

        pred = np.array(pred)
        return pred

    def get_pa_prediction(self, state, action):
        pa_prediction = self.model_prediction(state, action)
        return pa_prediction

    def get_pa_ratio(self, state, action1, action2):
        pa_prediction1 = self.model_prediction(state, action1)
        pa_prediction2 = self.model_prediction(state, action2)
        pa_ratio = pa_prediction1 / pa_prediction2
        return pa_ratio

    def get_a_prediction(self, state):
        if self.discrete_state:
            state = self.ohe_state.transform(state)
        if state.ndim == 1:
            state = state.reshape(1, state.size)
        # x = self.poly.fit_transform(state)
        x = self.rbf_feature.transform(state)
        action_pred = self.model.predict(x)
        return action_pred
