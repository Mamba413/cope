# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit
from opeuc import opeuc_run
from simulator_save import Simulator
from policy import target_policy
from utilize_ci import compute_ci, cover_truth

state_dim = 3
gamma = 0.9

# import logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(message)s', filename='DIS.log'.format(state_dim))

simulator = Simulator(dim_state=state_dim, model_type='save')

# palearner_setting = {'discrete_state': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
# pmlearner_setting = {'discrete_state': False, 'discrete_action': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
# qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [5], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
# ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': [5], 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}
palearner_setting = {'discrete_state': False,
                     'rbf_dim': state_dim, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting = {'discrete_state': False, 'discrete_action': False,
                     'rbf_dim': state_dim + 1, 'cv_score': 'accuracy', 'verbose': True}
# qlearner_setting = {'epoch': 200, 'trace': False, 'rbf_dim': (state_dim + 2)*2,
#                     'verbose': True, 'model': 'linear', 'eps': 1e-8} # add precision 
qlearner_setting = {'epoch': 500, 'trace': False, 'rbf_dim': (state_dim + 2)*6,
                    'verbose': True, 'model': 'linear', 'eps': 1e-8}
ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': state_dim * 3,
                        'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}

# opeuc_truth = simulator.estimate_ope(target_policy, gamma=gamma, mc_s0_time=80000, max_time=8000)
# print(opeuc_truth)
# opeuc_truth = 1.4681278408127834  # dimension: 1
# opeuc_truth = 1.8954356745210692  # dimension: 3 (s0: 50000, time: 5000)
opeuc_truth = 1.8934765607096165  # dimension: 3 (s0: 80000, time: 8000)

sim_rep = 500
num_time = 20
# num_trajectory_list = [20, 40, 80, 160, 320, 640, 1280]
# num_time_list = [20, 40, 80, 160, 320, 640, 1280]
# num_trajectory_list = [20, 40, 80, 160, 320]
# num_trajectory_list = [160, 320]
# num_trajectory_list = [640, 1280]
num_trajectory_list = [1280]
num_time_list = [5, 10, 20, 40, 80]
seed_list = list(range(sim_rep))
num_trajectory_list_size = len(num_trajectory_list)

cope_error = np.zeros((num_trajectory_list_size, sim_rep))
is_error = np.zeros((num_trajectory_list_size, sim_rep))
direct_error = np.zeros((num_trajectory_list_size, sim_rep))

cope_cover = np.zeros((num_trajectory_list_size, sim_rep))
is_cover = np.zeros((num_trajectory_list_size, sim_rep))
direct_cover = np.zeros((num_trajectory_list_size, sim_rep))


##################################################################
##################### Trajectory comparison ######################
##################################################################
for i, num_trajectory in enumerate(num_trajectory_list):
    # logging.info("Trajectory: {0} start.".format(num_trajectory))
    for k, r in enumerate(seed_list):
        if r % 10 == 0:
            print("Remain: ", 100 * (sim_rep - r) / sim_rep, "%.")
        pass

        simulator.sample_trajectory(num_trajectory, num_time, seed=r)
        simulator.trajectory2iid()
        sim_iid_dataset = simulator.iid_dataset
        s0 = sim_iid_dataset['s0']
        iid_dataset = []
        iid_dataset.append(sim_iid_dataset['state'])
        iid_dataset.append(sim_iid_dataset['action'])
        iid_dataset.append(sim_iid_dataset['mediator'])
        iid_dataset.append(sim_iid_dataset['reward'])
        iid_dataset.append(sim_iid_dataset['next_state'])
        
        ## trl:
        opeuc_obj = opeuc_run(s0, iid_dataset, target_policy, palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting, 
                                qlearner_setting = qlearner_setting, ratiolearner_setting=ratiolearner_setting)
        cope_est = opeuc_obj.opeuc
        is_est = opeuc_obj.cis
        direct_est = opeuc_obj.intercept
        cope_error[i, k] = cope_est - opeuc_truth
        is_error[i, k] = is_est - opeuc_truth
        direct_error[i, k] = direct_est - opeuc_truth

        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj.eif_arr)
        cope_cover[i, k] = cover_truth(opeuc_ci, opeuc_truth)
        is_ci = compute_ci(num_trajectory, num_time, opeuc_obj.cis_arr)
        is_cover[i, k] = cover_truth(is_ci, opeuc_truth)
        direct_ci = compute_ci(num_trajectory, 1, opeuc_obj.intercept_arr)
        direct_cover[i, k] = cover_truth(direct_ci, opeuc_truth)

        ### logging: 
        # logging.info("COPE: {0} CI-Cover: {1}".format(cope_error[i, k], cope_cover[i, k]))
        # logging.info("IS: {0} CI-Cover: {1}".format(is_error[i, k], is_cover[i, k]))
        # logging.info("Direct: {0} CI-Cover: {1}".format(direct_error[i, k], direct_cover[i, k]))
        pass
    pass


np.savetxt('trajectory-dic-cope-{0}.csv'.format(state_dim), cope_error, delimiter=", ")
np.savetxt('trajectory-dic-is-{0}.csv'.format(state_dim), is_error, delimiter=", ")
np.savetxt('trajectory-dic-direct-{0}.csv'.format(state_dim), direct_error, delimiter=", ")

np.savetxt('trajectory-dic-cope-cover-{0}.csv'.format(state_dim), cope_cover, delimiter=", ")
np.savetxt('trajectory-dic-is-cover-{0}.csv'.format(state_dim), is_cover, delimiter=", ")
np.savetxt('trajectory-dic-direct-cover-{0}.csv'.format(state_dim), direct_cover, delimiter=", ")

#############################################################
###################### Time comparison ######################
#############################################################
# num_trajectory = 80
# for i, num_time in enumerate(num_time_list):
#     logging.info("Time: {0} start.".format(num_time))
#     for k, r in enumerate(seed_list):
#         if r % 10 == 0:
#             print("Remain: ", 100 * (sim_rep - r) / sim_rep, "%.")
#         pass

#         simulator.sample_trajectory(num_trajectory, num_time, seed=r)
#         simulator.trajectory2iid()
#         sim_iid_dataset = simulator.iid_dataset
#         s0 = sim_iid_dataset['s0']
#         iid_dataset = []
#         iid_dataset.append(sim_iid_dataset['state'])
#         iid_dataset.append(sim_iid_dataset['action'])
#         iid_dataset.append(sim_iid_dataset['mediator'])
#         iid_dataset.append(sim_iid_dataset['reward'])
#         iid_dataset.append(sim_iid_dataset['next_state'])

#         ## trl:
#         opeuc_obj = opeuc_run(s0, iid_dataset, target_policy, palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting,
#                               qlearner_setting=qlearner_setting, ratiolearner_setting=ratiolearner_setting)
#         cope_est = opeuc_obj.opeuc
#         is_est = opeuc_obj.cis
#         direct_est = opeuc_obj.intercept
#         cope_error[i, k] = cope_est - opeuc_truth
#         is_error[i, k] = is_est - opeuc_truth
#         direct_error[i, k] = direct_est - opeuc_truth

#         opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj.eif_arr)
#         cope_cover[i, k] = cover_truth(opeuc_ci, opeuc_truth)
#         is_ci = compute_ci(num_trajectory, num_time, opeuc_obj.cis_arr)
#         is_cover[i, k] = cover_truth(is_ci, opeuc_truth)
#         direct_ci = compute_ci(num_trajectory, 1, opeuc_obj.intercept_arr)
#         direct_cover[i, k] = cover_truth(direct_ci, opeuc_truth)

#         ### logging:
#         logging.info("COPE: {0} CI-Cover: {1}".format(cope_error[i, k], cope_cover[i, k]))
#         logging.info("IS: {0} CI-Cover: {1}".format(is_error[i, k], is_cover[i, k]))
#         logging.info("Direct: {0} CI-Cover: {1}".format(direct_error[i, k], direct_cover[i, k]))
#         pass
#     pass


# np.savetxt('time-dic-cope-{0}.csv'.format(state_dim), cope_error, delimiter=", ")
# np.savetxt('time-dic-is-{0}.csv'.format(state_dim), is_error, delimiter=", ")
# np.savetxt('time-dic-direct-{0}.csv'.format(state_dim), direct_error, delimiter=", ")

# np.savetxt('time-dic-cope-cover-{0}.csv'.format(state_dim), cope_cover, delimiter=", ")
# np.savetxt('time-dic-is-cover-{0}.csv'.format(state_dim), is_cover, delimiter=", ")
# np.savetxt('time-dic-direct-cover-{0}.csv'.format(state_dim), direct_cover, delimiter=", ")
