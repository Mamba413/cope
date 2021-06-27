import numpy as np

class QModel2:
    def __init__(self, func):
        self.q_model = func
        pass

    def get_q_prediction(self, state, action):
        return self.q_model(state, action)

class QModel:
    def __init__(self, func, seed = 1):
        self.q_model = func
        self.seed = seed
        pass

    def get_q_prediction(self, state, action, mediator):
        np.random.seed(self.seed)
        return self.q_model(state, action, mediator)


class PAModel:
    def __init__(self, func, seed = 1):
        self.pa_model = func
        self.seed = seed
        pass

    def get_pa_prediction(self, state, action):
        np.random.seed(self.seed)
        return self.pa_model(state, action)

    def get_pa_ratio(self, state, policy_action, action):
        pa1 = self.get_pa_prediction(state, policy_action)
        pa2 = self.get_pa_prediction(state, action)
        return pa1 / pa2


class PMModel:
    def __init__(self, func, seed = 1, ratio_noise=False, false_func=None):
        self.pm_model = func
        self.ratio_noise = ratio_noise
        self.false_model = false_func
        self.seed = seed
        pass

    def get_pm_prediction(self, state, action, mediator):
        np.random.seed(self.seed)
        if self.false_model is None:
            pm_prediction = self.pm_model(state, action, mediator)
        else:
            pm_prediction = self.false_model(state, action, mediator)
        return pm_prediction

    def get_pm_ratio(self, state, policy_action, action, mediator):
        pm1 = self.get_pm_prediction(state, policy_action, mediator)
        pm2 = self.get_pm_prediction(state, action, mediator)
        pm_ratio = pm1 / pm2
        if self.ratio_noise:
            pm_ratio += np.random.normal(scale=0.05, size=pm_ratio.shape[0])
            pm_ratio = np.clip(pm_ratio, a_min=0.01, a_max=100)
        return pm_ratio

class RatioModel:
    def __init__(self, func, seed=1):
        self.r_model = func
        self.seed = seed
        pass

    def get_r_prediction(self, state):
        np.random.seed(self.seed)
        return self.r_model(state)
