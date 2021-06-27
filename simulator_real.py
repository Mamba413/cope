import numpy as np
from scipy.special import expit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from multiprocessing import Pool

class Simulator:
    def rideshare_init_state(self):
        # dim = 10
        # gaussian_mean = [0.9738, 1.8564, 6.0437, 1.1051,
        #                 1.1690, 4.0936, 1.6352, 0.5564, 1.8121, 2.4834]
        # gaussian_cov = [
        #     [0.128, -0.281, -0.599, 0.152, -0.238, -
        #         0.688, 0.242, -0.165, -0.513, -0.005],
        #     [1.358, 2.149, -0.317, 0.885, 1.919, -0.466, 0.424, 1.082, 0.019],
        #     [11.683, -0.629, 1.381, 7.236, -0.86, 0.672, 2.775, 0.427],
        #     [0.183, -0.277, -0.777, 0.295, -0.201, -0.623, -0.006],
        #     [0.749, 1.694, -0.406, 0.365, 0.949, 0.012],
        #     [7.813, -1.059, 0.819, 3.123, 0.285],
        #     [0.497, -0.329, -1.008, -0.008],
        #     [0.283, 0.773, 0.007],
        #     [2.721, 0.105],
        #     [0.769]
        # ]
        # cov_matrix = np.zeros((dim, dim))
        # iu1 = np.triu_indices(dim)
        # cov_matrix[iu1] = [i for item in gaussian_cov for i in item]
        # diag_cov_matrix = np.copy(np.diag(cov_matrix))
        # cov_matrix += cov_matrix.transpose()
        # cov_matrix -= np.diagflat(diag_cov_matrix)
        gaussian_mean = self.s0[0]
        cov_matrix = self.s0[1]
        init_state = np.random.multivariate_normal(gaussian_mean, cov_matrix, size=1).reshape(1, -1)
        return init_state
    
    def rideshare_confounder_model(self):
        return 0

    def rideshare_sc2action_model(self, state, confounder=0.0):
        pa = self.pa.predict_proba(state)[0, 0]
        return pa
    
    def rideshare_sa2mediator_model(self, state, action):
        sa = np.hstack([state, action.reshape(-1, 1)])
        pm = self.pm.predict_proba(sa)[0, 0]
        return pm

    def rideshare_smc2reward_model(self, state, mediator, action, random):
        sam = np.hstack(
            [state, action.reshape(-1, 1), mediator.reshape(-1, 1)])
        sam = self.pr_feature.fit_transform(sam)
        rmean = self.pr.predict(sam).reshape(-1)
        if random:
            reward = np.random.normal(size=1, loc=rmean, scale=np.sqrt(self.pr_noise_var))
        else:
            reward = rmean
        return reward

    def rideshare_smc2nextstate_model(self, state, mediator, action):
        sam = np.hstack(
            [state, action.reshape(-1, 1), mediator.reshape(-1, 1)])
        sam = self.pns_feature.fit_transform(sam)
        ns = self.pns.predict(sam)
        ns = np.random.multivariate_normal(ns.flatten(), self.pns_noise_var, size=1).reshape(1, -1)
        return ns
    
    def rideshare_smc2timegap_model(self, state, mediator, action):
        sam = np.hstack(
            [state, action.reshape(-1, 1), mediator.reshape(-1, 1)])
        sam = self.pt_feature.fit_transform(sam)
        time_gap = self.pt.predict(sam)
        return time_gap

    def __init__(self, s0, pa, pm, pr, pns, pt):
        self.dim_state = s0[0].shape[0]

        self.s0 = s0
        self.pa = pa
        self.pm = pm
        
        self.pr = pr['estimator']
        self.pr_feature = PolynomialFeatures(
            include_bias=False, degree=pr['params']['poly_features__degree'], interaction_only=pr['params']['poly_features__interaction_only'])
        self.pr_noise_var = pr['noise']

        self.pns = pns['estimator']
        self.pns_feature = PolynomialFeatures(
            include_bias=False, degree=pns['params']['poly_features__degree'], interaction_only=pns['params']['poly_features__interaction_only'])
        self.pns_noise_var = pns['noise']

        self.pt = pt['estimator']
        self.pt_feature = PolynomialFeatures(
            include_bias=False, degree=pt['params']['poly_features__degree'], interaction_only=pt['params']['poly_features__interaction_only'])

        self.init_state_model = self.rideshare_init_state
        self.confounder_model = self.rideshare_confounder_model
        self.sc2action_model = self.rideshare_sc2action_model
        self.sa2mediator_model = self.rideshare_sa2mediator_model
        self.smc2reward_model = self.rideshare_smc2reward_model
        self.smc2nextstate_model = self.rideshare_smc2nextstate_model
        self.smc2timegap_model = self.rideshare_smc2timegap_model

        self.trajectory_list = []
        self.target_policy_trajectory_list = []
        self.target_policy_state_density_list = None
        self.stationary_behaviour_policy_state_density = None
        pass

    def sample_init_state(self):
        init_state = self.init_state_model()
        return init_state

    def sample_confounder(self):
        confounder = self.confounder_model()
        return confounder

    def logistic_sampler(self, prob):
        if prob.ndim == 1:
            prob = prob[0]
        elif prob.ndim == 2:
            prob = prob[0][0]
        prob_arr = np.array([1-prob, prob])
        random_y = np.random.choice([0, 1], 1, p=prob_arr)
        return random_y

    def sample_sc2action(self, state, confounder, random=True):
        '''
        Output: a random action
        '''
        if random:
            random_action = self.logistic_sampler(
                self.sc2action_model(state, confounder))
        else:
            random_action = self.sc2action_model(state, confounder)
        return random_action

    def sample_sa2mediator(self, state, action):
        '''
        Output: a random mediator
        '''
        random_mediator = self.logistic_sampler(
            self.sa2mediator_model(state, action))
        return random_mediator

    def sample_smc2reward(self, state, mediator, confounder, random=True):
        random_reward = self.smc2reward_model(
            state, mediator, confounder, random=random)
        return random_reward

    def sample_smc2nextstate(self, state, mediator, confounder):
        random_next_state = self.smc2nextstate_model(state, mediator, confounder)
        return random_next_state

    def sample_smc2timegap(self, state, mediator, confounder):
        random_timegap = self.smc2timegap_model(state, mediator, confounder)
        return random_timegap

    def sample_one_trajectory(self, num_time, burn_in):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        if burn_in:
            burn_in_time = 50
            num_time += burn_in_time
        
        success_flag = True

        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_confounder = np.zeros(num_time)
        random_mediator = np.zeros(num_time)
        random_reward = np.zeros(num_time)
        random_timegap = np.zeros(num_time)

        random_state[0] = init_state.reshape(-1)

        try:
            for i in range(num_time):
                random_confounder[i] = self.sample_confounder()
                random_state_cp = np.copy(random_state[i].reshape(1, -1))
                random_action[i] = self.sample_sc2action(random_state_cp, np.copy(random_confounder[i]))
                random_action_cp = np.copy(random_action[i])
                random_mediator[i] = self.sample_sa2mediator(random_state_cp, np.copy(random_action[i]))
                random_mediator_cp = np.copy(random_mediator[i])

                random_reward[i] = self.sample_smc2reward(random_state_cp, random_mediator_cp, random_action_cp)

                if np.abs(random_reward[i]) > 1e5:
                    success_flag = False
                    pass

                random_state[i+1] = self.sample_smc2nextstate(random_state_cp, random_mediator_cp, random_action_cp)
                random_timegap[i] = self.sample_smc2timegap(random_state_cp, random_mediator_cp, random_action_cp)
                pass
        except ValueError:
            success_flag = False
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action,
                             random_mediator, random_reward, random_timegap]
        
        return random_trajectory, success_flag
    
    def sample_trajectory(self, num_trajectory, num_time, seed, burn_in=False, return_trajectory=False):
        tmp_list = self.trajectory_list.copy()
        self.trajectory_list = []
        np.random.seed(7654321*seed)
        while len(self.trajectory_list) < num_trajectory:
            one_trajectory, success_flag = self.sample_one_trajectory(num_time, burn_in)
            if success_flag:
                self.trajectory_list.append(one_trajectory)
            pass

        if return_trajectory:
            to_return_list = self.trajectory_list.copy()
            self.trajectory_list = tmp_list
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
                iid_dataset[5] = np.vstack([iid_dataset[5], iid_data[5]])
                iid_dataset[1] = np.append(iid_dataset[1], iid_data[1])
                iid_dataset[2] = np.append(iid_dataset[2], iid_data[2])
                iid_dataset[3] = np.append(iid_dataset[3], iid_data[3])
                iid_dataset[4] = np.append(iid_dataset[4], iid_data[4])
                pass
            pass

        self.iid_dataset = {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[5], 
                            'timegap': iid_dataset[4]}
        if trajectory is not None:
            return  {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[5],
                            'timegap': iid_dataset[4]}

    def estimate_ope(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_mediator_time=20, burn_in=False, seed=1, verbose=False, timegap=False):
        v_value_array = []
        j = 0
        if burn_in:
            burn_in_num = 50
            max_time += burn_in_num

        while len(v_value_array) < mc_s0_time:    
            np.random.seed(j+seed)
            current_state = self.sample_init_state()
            current_state = current_state.reshape(1, -1)
            v_value = 0.0
            observed_time = 0.0
            try:
                success_flag = True
                for i in range(max_time):
                    target_action = target_policy(np.copy(current_state).flatten())
                    action = self.sample_sc2action(np.copy(current_state), np.array([1.0]))
                    mediator = self.sample_sa2mediator(
                        np.copy(current_state), np.copy(target_action))
                    reward = self.sample_smc2reward(
                        np.copy(current_state), np.copy(mediator), np.copy(action), False)
                    # print(reward)
                    ## discard the diverged case:
                    if np.abs(reward) >= 1e4:
                        success_flag = False
                        break

                    if burn_in:
                        if i >= burn_in_num:
                            v_value += np.power(gamma, i-burn_in_num) * reward
                    elif timegap:
                        v_value += np.power(gamma, observed_time) * reward
                    else:
                        v_value += np.power(gamma, i) * reward
                        pass

                    time_gap = self.sample_smc2timegap(np.copy(current_state), np.copy(mediator), np.copy(action))[0]
                    observed_time += time_gap
                    # print(observed_time)
                    current_state = self.sample_smc2nextstate(np.copy(current_state), np.copy(mediator), np.copy(action))
                    pass
                if success_flag:
                    v_value = v_value[0]
                    v_value_array.append(v_value)
                pass
            except ValueError:
                pass

            j = j + 1
            pass

        if verbose:
            print(v_value_array)
            pass

        v_value_array = np.array(v_value_array)
        v_value_array = v_value_array[v_value_array > 0.0]
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

    def mc_ope(self, j, max_time, gamma, target_policy, seed, burn_in, timegap):
        if burn_in:
            burn_in_num = 50
            max_time += burn_in_num
            pass

        np.random.seed(j+seed)
        current_state = self.sample_init_state()
        current_state = current_state.reshape(1, -1)
        v_value = 0.0
        observed_time = 0.0
        try:
            success_flag = True
            for i in range(max_time):
                target_action = target_policy(np.copy(current_state).flatten())
                action = self.sample_sc2action(np.copy(current_state), np.array([1.0]))
                mediator = self.sample_sa2mediator(np.copy(current_state), np.copy(target_action))
                reward = self.sample_smc2reward(np.copy(current_state), np.copy(mediator), np.copy(action), False)
                # print(reward)
                ## discard the diverged case:
                if np.abs(reward) >= 1e4:
                    success_flag = False
                    break

                if burn_in:
                    if i >= burn_in_num:
                        v_value += np.power(gamma, i-burn_in_num) * reward
                elif timegap:
                    v_value += np.power(gamma, observed_time) * reward
                else:
                    v_value += np.power(gamma, i) * reward
                    pass

                time_gap = self.sample_smc2timegap(np.copy(current_state), np.copy(mediator), np.copy(action))[0]
                observed_time += time_gap
                # print(observed_time)
                current_state = self.sample_smc2nextstate(np.copy(current_state), np.copy(mediator), np.copy(action))
                pass
            if success_flag:
                v_value = v_value[0]
            pass
        except ValueError:
            pass
        return "{u},{v}".format(u=v_value, v=success_flag)

    def estimate_ope_parallel(self, target_policy, gamma, max_time=43, mc_s0_time=25, mc_mediator_time=20, burn_in=False, seed=1, verbose=False, timegap=False, num_process=5):
        if burn_in:
            burn_in_num = 50
            max_time += burn_in_num
        
        seed_list = np.arange(mc_s0_time, dtype='int64').tolist()
        max_time_list = (np.ones(mc_s0_time, dtype='int64') * max_time).tolist()
        gamma_list = (np.ones(mc_s0_time) * gamma).tolist()
        target_policy_list = [target_policy for i in range(mc_s0_time)]
        burn_in_list = [burn_in for i in range(mc_s0_time)]
        seed_offset = (np.ones(mc_s0_time, dtype='int64') * seed).tolist()
        timegap_list = np.array([timegap for i in range(mc_s0_time)])
        param_list = zip(seed_list, max_time_list, gamma_list,
                         target_policy_list, seed_offset, burn_in_list, 
                         timegap_list)
        
        with Pool(num_process) as p:
            v_value_array_res = p.starmap(self.mc_ope, param_list)
            pass

        v_value_array = np.array([a.split(',')[0] for a in v_value_array_res]).astype(np.float64)
        success_flag_array = np.array([a.split(',')[1] == 'True' for a in v_value_array_res])

        v_value_array = v_value_array[success_flag_array]
        if verbose:
            print(np.mean(success_flag_array))
            print(v_value_array[range(6)])
            pass

        true_v_value = np.mean(v_value_array)
        return true_v_value
