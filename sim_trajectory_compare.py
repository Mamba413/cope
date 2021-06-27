# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import expit
from opeuc import opeuc_run
from opedr import opedr_run
from simulator_save import Simulator
from policy import target_policy
from utilize_ci import compute_ci, cover_truth

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='trajectory_dim_1.log')

state_dim = 1
simulator = Simulator(dim_state=state_dim, model_type='save')
gamma = 0.9

palearner_setting={'discrete_state': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting={'discrete_state': False, 'discrete_action': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [5], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
ratiolearner_setting={'mode':'linear', 'rbf_ndims': [5], 'batch_size':32, 'epoch':3, 'lr':0.01, 'verbose': True}
wm_palearner_setting = {'discrete_state': False, 'rbf_dim': 4, 'cv_score': 'accuracy', 'verbose': True}
wm_qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [10], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
wm_ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': [10], 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}
# rbf_dim1 = [1, 2, 3, 4, 5]
# rbf_dim2 = [1, 3, 5, 7, 9]
# palearner_setting = {'discrete_state': False, 'rbf_dim': rbf_dim1, 'cv_score': 'accuracy', 'verbose': True}
# pmlearner_setting = {'discrete_state': False, 'discrete_action': False, 'rbf_dim': rbf_dim1, 'cv_score': 'accuracy', 'verbose': True}
# qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': rbf_dim2, 'verbose': True, 'model': 'linear', 'eps': 1e-5}
# ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': rbf_dim2, 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}
# wm_palearner_setting = {'discrete_state': False, 'rbf_dim': rbf_dim1, 'cv_score': 'accuracy', 'verbose': True}
# wm_qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': rbf_dim2, 'verbose': True, 'model': 'linear', 'eps': 1e-5}
# wm_ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': rbf_dim2, 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}

# opeuc_truth = simulator.estimate_ope(target_policy, gamma=gamma, mc_s0_time=500, max_time=500)
opeuc_truth = 1.4498987620607517
# logging.info("Truth: {0}".format(opeuc_truth))

sim_rep = 100
# num_trajectory_list = [60, 100, 140, 180, 220]
num_trajectory_list = [20, 40, 80, 160, 320]
num_time = 20
seed_list = list(range(sim_rep))
num_trajectory_list_size = len(num_trajectory_list)

reg_error = np.zeros((num_trajectory_list_size, sim_rep))
reg_error_wm = np.zeros((num_trajectory_list_size, sim_rep))
opedr_error = np.zeros((num_trajectory_list_size, sim_rep))
opedr_error_wm = np.zeros((num_trajectory_list_size, sim_rep))
is_opedr_error = np.zeros((num_trajectory_list_size, sim_rep))
is_opedr_error_wm = np.zeros((num_trajectory_list_size, sim_rep))
opeuc_error = np.zeros((num_trajectory_list_size, sim_rep))

opeuc_cover = np.zeros((num_trajectory_list_size, sim_rep))
opedr_cover = np.zeros((num_trajectory_list_size, sim_rep))
reg_cover = np.zeros((num_trajectory_list_size, sim_rep))
is_opedr_cover = np.zeros((num_trajectory_list_size, sim_rep))
opedr_wm_cover = np.zeros((num_trajectory_list_size, sim_rep))
reg_wm_cover = np.zeros((num_trajectory_list_size, sim_rep))
is_opedr_wm_cover = np.zeros((num_trajectory_list_size, sim_rep))

for i, num_trajectory in enumerate(num_trajectory_list):
    logging.info("Trajectory: {0} start.".format(num_trajectory))
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
        opeuc_est = 0.0
        opeuc_baseline_est = 0.0
        opeuc_obj = opeuc_run(s0, iid_dataset, target_policy, palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting, 
                                qlearner_setting = qlearner_setting, ratiolearner_setting=ratiolearner_setting)
        opeuc_est = opeuc_obj.opeuc
        opeuc_error[i, k] = opeuc_est - opeuc_truth
        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj.eif_arr)
        opeuc_cover[i, k] = cover_truth(opeuc_ci, opeuc_truth)

        ## REG, IS, DRL
        opedr_est = 0.0
        opedr_baseline_est = 0.0
        is_est = 0.0
        opedr_obj = opedr_run(s0, iid_dataset, target_policy, gamma=gamma, palearner_setting=palearner_setting,
                                qlearner_setting=qlearner_setting, ratiolearner_setting=ratiolearner_setting, matrix_based_ope=True)
        opedr_est = opedr_obj.opedr
        opedr_baseline_est = opedr_obj.intercept
        is_est = opedr_obj.weight_reward
        opedr_error[i, k] = opedr_est - opeuc_truth
        reg_error[i, k] = opedr_baseline_est - opeuc_truth
        is_opedr_error[i, k] = is_est - opeuc_truth

        opedr_ci = compute_ci(num_trajectory, num_time, opedr_obj.eif_arr)
        opedr_cover[i, k] = cover_truth(opedr_ci, opeuc_truth)
        reg_ci = compute_ci(num_trajectory, 1, opedr_obj.intercept_arr)
        reg_cover[i, k] = cover_truth(reg_ci, opeuc_truth)
        is_ci = compute_ci(num_trajectory, num_time, opedr_obj.is_arr)
        is_opedr_cover[i, k] = cover_truth(is_ci, opeuc_truth)

        ## REG, IS, DRL (with mediator)
        mediator_mat = iid_dataset[2].reshape(num_trajectory, num_time)
        mediator_mat = np.hstack([mediator_mat, np.random.binomial(n=1, p=0.5, size=s0.shape[0]).reshape(-1, 1)])
        mediator_mat1 = mediator_mat[:, 0:num_time]
        mediator_mat1 = mediator_mat1.reshape(-1, 1)
        mediator_mat2 = mediator_mat[:, 1:]
        mediator_mat2 = mediator_mat2.reshape(-1, 1)

        s0 = np.hstack([s0, mediator_mat[:, 0].reshape(-1, 1)])
        iid_dataset[0] = np.hstack([iid_dataset[0], mediator_mat1])
        iid_dataset[4] = np.hstack([iid_dataset[4], mediator_mat2])

        opedr_obj2 = opedr_run(s0, iid_dataset, target_policy, gamma=gamma, palearner_setting=wm_palearner_setting,
                               qlearner_setting=wm_qlearner_setting, ratiolearner_setting=wm_ratiolearner_setting, matrix_based_ope=True)
        opedr_est_wm = opedr_obj2.opedr
        opedr_baseline_est_wm = opedr_obj2.intercept
        is_est_wm = opedr_obj2.weight_reward

        opedr_error_wm[i, k] = opedr_est_wm - opeuc_truth
        reg_error_wm[i, k] = opedr_baseline_est_wm - opeuc_truth
        is_opedr_error_wm[i, k] = is_est_wm - opeuc_truth

        opedr_wm_ci = compute_ci(num_trajectory, num_time, opedr_obj2.eif_arr)
        opedr_wm_cover[i, k] = cover_truth(opedr_ci, opeuc_truth)
        reg_wm_ci = compute_ci(num_trajectory, 1, opedr_obj2.intercept_arr)
        reg_wm_cover[i, k] = cover_truth(reg_ci, opeuc_truth)
        is_wm_ci = compute_ci(num_trajectory, num_time, opedr_obj2.is_arr)
        is_opedr_wm_cover[i, k] = cover_truth(is_ci, opeuc_truth)

        logging.info("COPE: {0}, CI-Cover: {1}".format(opeuc_error[i, k], opeuc_cover[i, k]))
        logging.info("DRL: {0}, CI-Cover: {1}".format(opedr_error[i, k], opedr_cover[i, k]))
        logging.info("REG: {0}, CI-Cover: {1}".format(reg_error[i, k], reg_cover[i, k]))
        logging.info("MIS: {0}, CI-Cover: {1}".format(is_opedr_error[i, k], is_opedr_cover[i, k]))
        logging.info("DRL-WM: {0}, CI-Cover: {1}".format(opedr_error_wm[i, k], opedr_wm_cover[i, k]))
        logging.info("REG-WM: {0}, CI-Cover: {1}".format(reg_error_wm[i, k], reg_wm_cover[i, k]))
        logging.info("MIS-WM: {0}, CI-Cover: {1}".format(is_opedr_error_wm[i, k], is_opedr_wm_cover[i, k]))
        pass
    pass

np.savetxt('trajectory-reg-1.csv', reg_error, delimiter=", ")
np.savetxt('trajectory-regwm-1.csv', reg_error_wm, delimiter=", ")
np.savetxt('trajectory-drl-1.csv', opedr_error, delimiter=", ")
np.savetxt('trajectory-drlwm-1.csv', opedr_error_wm, delimiter=", ")
np.savetxt('trajectory-is-1.csv', is_opedr_error, delimiter=", ")
np.savetxt('trajectory-iswm-1.csv', is_opedr_error_wm, delimiter=", ")
np.savetxt('trajectory-trl-1.csv', opeuc_error, delimiter=", ")

np.savetxt('trajectory-reg-cover-1.csv'.format(state_dim), reg_cover, delimiter=", ")
np.savetxt('trajectory-regwm-cover-1.csv'.format(state_dim), reg_wm_cover, delimiter=", ")
np.savetxt('trajectory-drl-cover-1.csv'.format(state_dim), opedr_cover, delimiter=", ")
np.savetxt('trajectory-drlwm-cover-1.csv'.format(state_dim), opedr_wm_cover, delimiter=", ")
np.savetxt('trajectory-is-cover-1.csv'.format(state_dim), is_opedr_cover, delimiter=", ")
np.savetxt('trajectory-iswm-cover-1.csv'.format(state_dim), is_opedr_wm_cover, delimiter=", ")
np.savetxt('trajectory-trl-cover-1.csv'.format(state_dim), opeuc_cover, delimiter=", ")
