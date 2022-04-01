import numpy as np
from scipy.special import expit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool

class Simulator:
    def toy_init_state(self):
        init_state = np.random.binomial(n=1, p=0.5, size=1).reshape(-1)
        return init_state
    
    def toy_confounder_model(self, state, last_state):
        pu = expit(0.5*np.sum(state) + 0.5*np.sum(last_state))
        confounder = 2 * np.random.binomial(n=1, p=pu, size=1) - 1
        return confounder
    
    def toy_sc2action_model(self, state, confounder):
        pa = 0.1*np.sum(state) + 0.9*confounder
        pa = expit(pa)
        pa = np.array([0.25*pa, 1-pa, 0.75*pa])
        return pa
    
    def toy_sa2mediator_model(self, state, action):
        pm = 0.1*np.sum(state) + 0.9*action - 0.45
        pm = expit(pm)
        return pm
    
    def toy_smc2reward_model(self, state, mediator, confounder, random):
        rmean = 0.5 * np.clip(confounder, 0, 1) * (state[0] + mediator) - 0.1 * state[0]
        rmean = expit(rmean)
        if random:
            reward = np.random.binomial(n=1, p=rmean, size=1)
        else:
            reward = rmean
        reward *= 10
        return reward
    
    def toy_smc2nextstate_model(self, state, mediator, confounder):
        next_state = 0.5 * np.clip(confounder, 0, 1) * (state[0] + mediator) - 0.1 * state[0]
        next_state = expit(next_state)
        next_state = np.random.binomial(n=1, p=next_state, size=1)
        return next_state

    def toy2_init_state(self):
        init_state = self.toy_init_state()
        return init_state
    
    def toy2_confounder_model(self, state, last_confounder):
        pu = expit(0.5 * state + 0.5 * last_confounder)
        confounder = 2 * np.random.binomial(n=1, p=pu, size=1) - 1
        return confounder
    
    def toy2_sc2action_model(self, state, confounder):
        pa = self.toy_sc2action_model(state, confounder)
        return pa
    
    def toy2_sa2mediator_model(self, state, action):
        pm = self.toy_sa2mediator_model(state, action)
        return pm
    
    def toy2_smc2reward_model(self, state, mediator, confounder, random):
        reward = self.toy_smc2reward_model(state, mediator, confounder, random)
        return reward
    
    def toy2_smc2nextstate_model(self, state, mediator, confounder):
        next_state = self.toy_smc2nextstate_model(state, mediator, confounder)
        return next_state

    def __init__(self, model_type='toy', dim_state=3):
        self.dim_state = dim_state
        if model_type == "toy":
            self.model_type = "toy"
            self.init_state_model = self.toy_init_state
            self.confounder_model = self.toy_confounder_model
            self.sc2action_model = self.toy_sc2action_model
            self.sa2mediator_model = self.toy_sa2mediator_model
            self.smc2reward_model = self.toy_smc2reward_model
            self.smc2nextstate_model = self.toy_smc2nextstate_model
        elif model_type == "toy2":
            self.model_type = "toy2"
            self.init_state_model = self.toy2_init_state
            self.confounder_model = self.toy2_confounder_model
            self.sc2action_model = self.toy2_sc2action_model
            self.sa2mediator_model = self.toy2_sa2mediator_model
            self.smc2reward_model = self.toy2_smc2reward_model
            self.smc2nextstate_model = self.toy2_smc2nextstate_model
            pass

        self.trajectory_list = []
        self.target_policy_trajectory_list = []
        self.target_policy_state_density_list = None
        self.stationary_behaviour_policy_state_density = None
        pass

    def sample_init_state(self):
        init_state = self.init_state_model()
        return init_state

    def sample_confounder(self, state, value):
        confounder = self.confounder_model(state, value)
        return confounder

    def logistic_sampler(self, prob):
        prob_size = np.array(prob).flatten().size
        if prob_size <= 2:
            if prob.ndim == 1:
                prob = prob[0]
            elif prob.ndim == 2:
                prob = prob[0][0]
            prob_arr = np.array([1-prob, prob])
            random_y = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            prob_arr = prob.flatten()
            random_y = np.random.choice([-1, 0, 1], 1, p=prob_arr)
        return random_y

    def sample_sc2action(self, state, confounder, random=True):
        '''
        Output: a random action
        '''
        if random:
            random_action = self.logistic_sampler(self.sc2action_model(state, confounder))
        else:
            random_action = self.sc2action_model(state, confounder)
        return random_action

    def sample_sa2mediator(self, state, action):
        '''
        Output: a random mediator
        '''
        random_mediator = self.logistic_sampler(self.sa2mediator_model(state, action))
        return random_mediator

    def sample_smc2reward(self, state, mediator, confounder, random=True):
        random_reward = self.smc2reward_model(state, mediator, confounder, random=random)
        return random_reward

    def sample_smc2nextstate(self, state, mediator, confounder):
        random_next_state = self.smc2nextstate_model(
            state, mediator, confounder)
        return random_next_state

    def sample_one_trajectory(self, num_time, burn_in):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        if burn_in:
            burn_in_time = 50
            num_time += burn_in_time

        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_confounder = np.zeros(num_time)
        random_mediator = np.zeros(num_time)
        random_reward = np.zeros(num_time)

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            if i == 0:
                value = np.array([0.0])
            else:
                if self.model_type == 'toy':
                    value = random_state[i-1]
                else:
                    value = random_confounder[i-1]
                    pass
                pass

            random_confounder[i] = self.sample_confounder(random_state[i], value)
            random_action[i] = self.sample_sc2action(
                random_state[i], random_confounder[i])
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], random_action[i])
            random_reward[i] = self.sample_smc2reward(
                random_state[i], random_mediator[i], random_confounder[i])
            random_state[i+1] = self.sample_smc2nextstate(
                random_state[i], random_mediator[i], random_confounder[i])
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action, random_mediator, random_reward]
        
        return random_trajectory
    
    def sample_trajectory(self, num_trajectory, num_time, seed, burn_in=False, return_trajectory=False):
        tmp_list = self.trajectory_list.copy()
        self.trajectory_list = []
        np.random.seed(7654321*seed)
        for i in range(num_trajectory):
            one_trajectory = self.sample_one_trajectory(num_time, burn_in)
            self.trajectory_list.append(one_trajectory)
            pass

        if return_trajectory:
            to_return_list = self.trajectory_list.copy()
            self.trajectory_list = tmp_list
            return to_return_list

    def sample_one_target_policy_trajectory(self, num_time, target_policy):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_confounder = np.zeros(num_time)
        random_mediator = np.zeros(num_time)
        random_reward = np.zeros(num_time)

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            if i == 0:
                last_state = 0.0
            else:
                last_state = random_state[i-1]
            random_confounder[i] = self.sample_confounder(random_state[i], last_state)
            random_action[i] = target_policy(random_state[i])
            # random_mediator[i] = self.sample_action2mediator(random_action[i])
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], random_action[i])
            random_reward[i] = self.sample_smc2reward(
                random_state[i], random_mediator[i], random_confounder[i])
            random_state[i+1] = self.sample_smc2nextstate(
                random_state[i], random_mediator[i], random_confounder[i])
            pass

        random_trajectory = [random_state, random_action,
                             random_mediator, random_reward]
        return random_trajectory

    def sample_target_policy_trajectory(self, num_trajectory, num_time, target_policy, seed, return_trajectory=False):
        tmp_list = self.target_policy_trajectory_list.copy()
        self.target_policy_trajectory_list = []
        np.random.seed(seed)
        for i in range(num_trajectory):
            one_trajectory = self.sample_one_target_policy_trajectory(
                num_time, target_policy)
            self.target_policy_trajectory_list.append(one_trajectory)
            pass

        if return_trajectory:
            to_return_list = self.target_policy_trajectory_list.copy()
            self.target_policy_trajectory_list = tmp_list
            return to_return_list

    def onetrajectory2iid(self, trajectory):
        num_time = trajectory[1].shape[0]
        s0 = trajectory[0][0]
        state = trajectory[0][range(num_time)]
        next_state = trajectory[0][range(1, num_time+1)]
        trajectory[0] = state
        trajectory.append(next_state)
        return s0, trajectory

    def trajectory2iid(self, trajectory=None):
        iid_dataset = []
        if trajectory is None:
            trajectory_list = self.trajectory_list.copy()
        else:
            trajectory_list = trajectory.copy()
            pass

        num_trajectory = len(trajectory_list)
        for i in range(num_trajectory):
            s0_data, iid_data = self.onetrajectory2iid(trajectory_list[i])
            if i == 0:
                iid_dataset = iid_data
                s0_dataset = s0_data
            else:
                s0_dataset = np.vstack([s0_dataset, s0_data])
                iid_dataset[0] = np.vstack([iid_dataset[0], iid_data[0]])
                iid_dataset[4] = np.vstack([iid_dataset[4], iid_data[4]])
                iid_dataset[1] = np.append(iid_dataset[1], iid_data[1])
                iid_dataset[2] = np.append(iid_dataset[2], iid_data[2])
                iid_dataset[3] = np.append(iid_dataset[3], iid_data[3])
                pass
            pass

        self.iid_dataset = {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[4]}
        if trajectory is not None:
            return  {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[4]}

    def estimate_ope(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_mediator_time=20, burn_in=False, seed=1):
        v_value_array = np.zeros(mc_s0_time)
        if burn_in:
            burn_in_num = 50
            max_time += burn_in_num
        for j in range(mc_s0_time):
            np.random.seed(j+seed)
            current_state = self.sample_init_state()
            current_state = current_state.reshape(-1)
            v_value = 0.0
            for i in range(max_time):
                if i == 0:
                    value = np.array([0.0])
                else:
                    if self.model_type == 'toy':
                        value = last_state.copy()
                    else:
                        value = confounder.copy()
                        pass
                    pass
                action = target_policy(current_state)
                confounder = self.sample_confounder(current_state, value)
                mediator = self.sample_sa2mediator(current_state, action)
                reward = self.sample_smc2reward(current_state, mediator, confounder, False)
                if burn_in:
                    if i >= burn_in_num:
                        v_value += np.power(gamma, i-burn_in_num) * reward
                else:
                    v_value += np.power(gamma, i) * reward
                last_state = current_state.copy()
                current_state = self.sample_smc2nextstate(current_state, mediator, confounder)
                pass
            v_value_array[j] = v_value
            pass

        true_v_value = np.mean(v_value_array)
        return true_v_value

    def mc_ope(self, j, max_time, gamma, target_policy, seed, burn_in):
        np.random.seed(j+seed)
        current_state = self.sample_init_state()
        current_state = current_state.reshape(-1)
        v_value = 0.0
        for i in range(max_time):
            action = target_policy(current_state)
            confounder = self.sample_confounder()
            mediator = self.sample_sa2mediator(current_state, action)
            reward = self.sample_smc2reward(
                current_state, mediator, confounder, False)
            if burn_in:
                if i >= burn_in_num:
                    v_value += np.power(gamma, i-burn_in_num) * reward
            else:
                v_value += np.power(gamma, i) * reward
            current_state = self.sample_smc2nextstate(
                current_state, mediator, confounder)
            pass
        return v_value

    def estimate_ope_parallel(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_mediator_time=20, burn_in=False, seed=1, num_process=5):
        if burn_in:
            burn_in_num = 50
            max_time += burn_in_num
        
        seed_list = np.arange(mc_s0_time, dtype='int64').tolist()
        seed_offset = (np.ones(mc_s0_time, dtype='int64') * seed).tolist()
        max_time_list = (np.ones(mc_s0_time, dtype='int64') * max_time).tolist()
        gamma_list = (np.ones(mc_s0_time) * gamma).tolist()
        target_policy_list = [target_policy for i in range(mc_s0_time)]
        burn_in_list = [burn_in for i in range(mc_s0_time)]
        param_list = zip(seed_list, max_time_list, gamma_list,
                         target_policy_list, seed_offset, burn_in_list)
        with Pool(num_process) as p:
            v_value_array = p.starmap(self.mc_ope, param_list)

        true_v_value = np.mean(v_value_array)
        return true_v_value

    def estimate_mediated_ope(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_mediator_time=20):
        v_value_array = np.zeros(mc_s0_time)
        for j in range(mc_s0_time):
            np.random.seed(j)
            current_state = self.sample_init_state()
            current_state = current_state.reshape(-1)
            v_value = 0.0
            for i in range(max_time):
                confounder = self.sample_confounder()
                action = self.sample_sc2action(current_state, confounder)
                mediator = self.sample_sa2mediator(
                    current_state, target_policy(current_state))
                reward = self.sample_smc2reward(
                    current_state, mediator, confounder, False)
                v_value += np.power(gamma, i) * reward
                current_state = self.sample_smc2nextstate(
                    current_state, mediator, confounder)
                pass
            v_value_array[j] = v_value
            pass
        true_v_value = np.mean(v_value_array)
        return true_v_value

    def estimate_vfunction_s0(self, s0, target_policy, gamma, max_time=43, mc_mediator_time=20):
        v_value_array = np.zeros(mc_mediator_time)
        for j in range(mc_mediator_time):
            np.random.seed(j)
            current_state = np.copy(s0)
            current_state = current_state.reshape(-1)
            v_value = 0.0
            for i in range(max_time):
                action = target_policy(current_state)
                confounder = self.sample_confounder()
                mediator = self.sample_sa2mediator(current_state, action)
                reward = self.sample_smc2reward(
                    current_state, mediator, confounder, random=False)
                v_value += np.power(gamma, i) * reward
                current_state = self.sample_smc2nextstate(
                    current_state, mediator, confounder)
                pass
            v_value_array[j] = v_value
            pass
        true_v_value = np.mean(v_value_array)
        return true_v_value

    def estimate_qfunction(self, reward, next_state, target_policy, gamma, max_time=43, mc_mediator_time=20):
        v_est = self.estimate_vfunction_s0(
            next_state, target_policy, gamma, max_time, mc_mediator_time)
        reward_est = reward
        q_function_est = v_est * gamma + reward_est
        return q_function_est

    def estimate_ope_via_qfunction(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_cp_action_time=200, mc_cp_mediator_time=200):
        ope_value_array = np.zeros(mc_s0_time)
        action_value = np.array([0.0, 1.0])
        action_prob_value = np.array([0.0, 0.0])
        mediator_value = np.array([0.0, 1.0])
        mediator_prob_value = np.array([0.0, 0.0])
        for j in range(mc_s0_time):
            np.random.seed(j)
            current_state = self.sample_init_state()
            current_state = current_state.reshape(-1)

            ## estimate p(a|s)
            mc_action_array = np.zeros(mc_cp_action_time)
            for r in range(mc_cp_action_time):
                np.random.seed(r)
                # confounder = self.sample_confounder()
                # mc_action_array[r] = self.sample_sc2action(
                #     current_state, confounder)
                mc_action_array[r] = target_policy(current_state)
                pass
            for r, one_action in enumerate(action_value):
                action_prob_value[r] = np.where(
                    mc_action_array == one_action)[0].shape[0]
                pass
            action_prob_value /= float(mc_cp_action_time)
            # print(action_prob_value)

            ## estimate p(m|s, \pi(s))
            mc_mediator_array = np.zeros(mc_cp_mediator_time)
            for r in range(mc_cp_mediator_time):
                np.random.seed(r)
                mc_mediator_array[r] = self.sample_sa2mediator(
                    current_state, target_policy(current_state))
                pass
            for r, one_mediator in enumerate(mediator_value):
                mediator_prob_value[r] = np.where(
                    mc_mediator_array == one_mediator)[0].shape[0]
                pass
            mediator_prob_value /= float(mc_cp_mediator_time)
            # print(mediator_prob_value)

            ## compute q value
            confounder = self.sample_confounder()
            q_value_list = []
            for s, one_action in enumerate(action_value):
                for t, one_mediator in enumerate(mediator_value):
                    reward = self.sample_smc2reward(
                        current_state, one_mediator, confounder, random=False)
                    next_state = self.sample_smc2nextstate(
                        current_state, one_mediator, confounder)
                    q_value = self.estimate_qfunction(
                        reward, next_state, target_policy, gamma=gamma, max_time=max_time)
                    q_value *= mediator_prob_value[s]
                    q_value *= action_prob_value[t]
                    q_value_list.append(q_value)
                    pass
                pass
            q_value_list = np.array(q_value_list)
            ope_value_array[j] = np.sum(q_value_list)
            pass
        true_ope = np.mean(ope_value_array)
        return true_ope

    def extract_state(self, trajectory_list, num_trajectory):
        state_list = []
        for i in range(num_trajectory):
            state_list.append(trajectory_list[i][0])
            pass
        state_all = np.vstack(state_list)
        return state_all

    def estimate_behaviour_policy_state_density(self, num_trajectory, num_time, seed):
        num_trajectory = len(self.trajectory_list)
        if num_trajectory == 0:
            self.sample_trajectory(num_trajectory, num_time, seed)
            pass
        state_all = self.extract_state(self.trajectory_list, num_trajectory)

        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(state_all)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        self.stationary_behaviour_policy_state_density = grid.best_estimator_
        pass

    def estimate_target_policy_state_density(self, num_trajectory, num_time, target_policy, seed):
        num_trajectory = len(self.target_policy_trajectory_list)
        if num_trajectory == 0:
            self.sample_target_policy_trajectory(
                num_trajectory, num_time, target_policy, seed)
            pass
        
        state_list = []
        for t in range(num_time):
            state_list_time_t = []
            for i in range(num_trajectory):
                state_list_time_t.append(self.target_policy_trajectory_list[i][0][t, :])
                pass
            print(state_list_time_t)
            t_state = np.vstack(state_list_time_t)
            state_list.append(t_state)

        params = {'bandwidth': np.logspace(-1, 1, 20)}
        for t in range(num_time):
            grid = GridSearchCV(KernelDensity(), params)
            grid.fit(state_list[t])
            print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
            self.target_policy_state_density_list[t] = grid.best_estimator_
            pass
        pass

    def estimate_ratio(self, state, gamma, num_trajectory, num_time, target_policy, max_time=43, seed=1, replace=False):
        if self.target_policy_state_density_list is None or replace:
            self.estimate_target_policy_state_density(num_trajectory, num_time, target_policy, seed)
        
        if self.stationary_behaviour_policy_state_density is None or replace:
            self.estimate_behaviour_policy_state_density(num_trajectory, num_time, seed)
            
        eval_num = state.shape[0]
        numerator = np.zeros(eval_num)
        for t in range(max_time):
            numerator += np.power(gamma, t) * self.target_policy_state_density_list[t].score_samples(state)
            pass
        denominator = self.stationary_behaviour_policy_state_density.score_samples(state)
        ratio_value = numerator / denominator
        return ratio_value

    def estimate_discrete_ratio(self, num_trajectory, num_time, target_policy, seed, burn_in=False):
        trajectory_discrete_ratio = self.sample_trajectory(
            num_trajectory, num_time, seed, burn_in=False, return_trajectory=True)
        all_state0 = [_trajectory[0] for _trajectory in trajectory_discrete_ratio]
        all_state0 = np.hstack(all_state0)
        self.obs_prob_array = np.apply_along_axis(np.mean, 1, all_state0)

        trajectory_discrete_ratio = self.sample_target_policy_trajectory(
            num_trajectory, num_time, target_policy, seed, return_trajectory=True)
        all_state = [_trajectory[0] for _trajectory in trajectory_discrete_ratio]
        all_state = np.hstack(all_state)
        all_state = all_state[:, -1]
        self.target_policy_stationary_prob = np.mean(all_state)

    def predict_discrete_ratio(self, state, gamma):
        denominator = self.target_policy_stationary_prob * state
        denominator += (1.0 - self.target_policy_stationary_prob) * (1.0 - state)
        numerator = np.array([0.0])
        for t, prob in enumerate(self.obs_prob_array):
            numerator_part = prob * state
            numerator_part += (1.0 - prob) * (1.0 - state)
            numerator_part *= np.power(gamma, t)
            numerator += numerator_part
            pass
        ratio = (1.0 - gamma) * numerator / denominator
        return ratio
