import numpy as np
from scipy.special import expit
from cope_inf import cope_inf_run
from simulator_inf import Simulator
from policy import target_policy_action3_inf as target_policy
from utilize_ci import compute_ci, cover_truth
from is_inf import is_inf_run
import copy

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='trajectory_inf_1.log')

state_dim = 1
simulator = Simulator(dim_state=state_dim, model_type='toy2')

palearner_setting = {'discrete_state': False,
                     'rbf_dim': 5, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting = {'discrete_state': False, 'discrete_action': False,
                     'rbf_dim': 5, 'cv_score': 'accuracy', 'verbose': True}
qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [5], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
cond_prob_learner_setting = {'discrete_state': False, 'rbf_dim': "raw", 'cv_score': 'accuracy', 'verbose': True}

gamma = 0.9
sim_rep = 200
# num_trajectory_list = [160, 320, 640]
# num_trajectory_list = [20, 40, 80, 160, 320, 640]
num_trajectory_list = [80, 160, 320, 640, 1280, 2560]
num_trajectory_list_size = len(num_trajectory_list)
num_time = 3

# opeuc_truth = simulator.estimate_ope(target_policy, gamma=gamma, mc_s0_time=30000, max_time=num_time)
# opeuc_truth = 5.763636  ## U_t <-- f(S_t, S_{t-1}), Manual calculation
# opeuc_truth = 15.271916157524167  ## U_t <-- f(S_t, S_{t-1}), Monte Carol, 30000 trajectory
opeuc_truth = 15.177699143512935  ## U_t <-- f(S_t, U_{t-1}), Monte Carol, 30000 trajectory
print(opeuc_truth)

seed_list = list(range(sim_rep))
q_error = np.zeros((num_trajectory_list_size, sim_rep))
q_cover = np.zeros((num_trajectory_list_size, sim_rep))
is_error = np.zeros((num_trajectory_list_size, sim_rep))
is_cover = np.zeros((num_trajectory_list_size, sim_rep))

for i, num_trajectory in enumerate(num_trajectory_list):
    logging.info("Trajectory: {0} start.".format(num_trajectory))
    for k, r in enumerate(seed_list):
        if r % 10 == 0:
            print("Remain: ", 100 * (sim_rep - r) / sim_rep, "%.")
        pass

        simulator.sample_trajectory(num_trajectory, num_time, seed=r)
        trajectory_dataset = copy.deepcopy(simulator.trajectory_list)  # simulator.trajectory_list will change after i.i.d. transformation
        # trajectory_dataset = simulator.trajectory_list.copy()        # .copy() is not enough
        simulator.trajectory2iid()
        sim_iid_dataset = simulator.iid_dataset
        s0 = sim_iid_dataset['s0']
        iid_dataset = []
        iid_dataset.append(sim_iid_dataset['state'])
        iid_dataset.append(sim_iid_dataset['action'])
        iid_dataset.append(sim_iid_dataset['mediator'])
        iid_dataset.append(sim_iid_dataset['reward'])
        iid_dataset.append(sim_iid_dataset['next_state'])

        q_obj = cope_inf_run(s0, iid_dataset, target_policy, num_trajectory, num_time,
                                 palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting,
                                 qlearner_setting=qlearner_setting)
        q_est = q_obj.opeuc
        q_error[i, k] = q_est - opeuc_truth
        q_ci = compute_ci(num_trajectory, 1, q_obj.eif_arr)
        q_cover[i, k] = cover_truth(q_ci, opeuc_truth)

        is_obj = is_inf_run(trajectory_dataset, target_policy, cond_prob_learner_setting=cond_prob_learner_setting)
        is_est = is_obj.opeuc
        is_error[i, k] = is_est - opeuc_truth
        is_ci = compute_ci(num_trajectory, 1, is_obj.eif_arr)
        is_cover[i, k] = cover_truth(is_ci, opeuc_truth)

        logging.info("Q-INF: {0}, CI-Cover: {1}".format(q_error[i, k], q_cover[i, k]))
        logging.info("IS-INF: {0}, CI-Cover: {1}".format(is_error[i, k], is_cover[i, k]))
        pass
    pass

np.savetxt('inf-trajectory-qinf-1.csv', q_error, delimiter=", ")
np.savetxt('inf-trajectory-qinf-cover-1.csv'.format(state_dim), q_cover, delimiter=", ")
np.savetxt('inf-trajectory-isinf-1.csv', is_error, delimiter=", ")
np.savetxt('inf-trajectory-isinf-cover-1.csv'.format(state_dim), is_cover, delimiter=", ")