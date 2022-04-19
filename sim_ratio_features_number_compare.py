import numpy as np
from scipy.special import expit
from opeuc import opeuc_run
from opedr import opedr_run
from simulator_save import Simulator
from policy import target_policy
from utilize_ci import compute_ci, cover_truth

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='ratio_feature.log')

state_dim = 3
simulator = Simulator(dim_state=state_dim, model_type='save')

if state_dim == 1:
    palearner_setting={'discrete_state': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
    pmlearner_setting={'discrete_state': False, 'discrete_action': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
    qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [5], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
    ratiolearner_setting={'mode':'linear', 'rbf_ndims': [5], 'batch_size':32, 'epoch':3, 'lr':0.01, 'verbose': True}
    wm_palearner_setting = {'discrete_state': False, 'rbf_dim': 4, 'cv_score': 'accuracy', 'verbose': True}
    wm_qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [10], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
    wm_ratiolearner_setting={'mode':'linear', 'rbf_ndims': [10], 'batch_size':32, 'epoch':3, 'lr':0.01, 'verbose': True}
    opeuc_truth = 1.4498987620607517
    ratiolearner_feature_number = [2, 4, 6, 8, 10]
elif state_dim == 3:
    palearner_setting = {'discrete_state': False, 'rbf_dim': state_dim, 'cv_score': 'accuracy', 'verbose': True}
    pmlearner_setting = {'discrete_state': False, 'discrete_action': False, 'rbf_dim': state_dim + 1, 'cv_score': 'accuracy', 'verbose': True}
    qlearner_setting = {'epoch': 500, 'trace': False, 'rbf_dim': (state_dim + 2)*6, 'verbose': True, 'model': 'linear', 'eps': 1e-8}
    ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': state_dim * 3, 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}
    wm_palearner_setting = {'discrete_state': False, 'rbf_dim': 5, 'cv_score': 'accuracy', 'verbose': True}
    wm_qlearner_setting = {'epoch': 500, 'trace': False, 'rbf_dim': (state_dim + 3)*6, 'verbose': True, 'model': 'linear', 'eps': 1e-8}
    wm_ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': 6, 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}
    opeuc_truth = 1.8934765607096165  
    ratiolearner_feature_number = [20, 25, 30, 35, 40]
    pass

gamma = 0.9
sim_rep = 200

num_trajectory = 160
num_time = 20
seed_list = list(range(sim_rep))
num_feature_list_size = len(ratiolearner_feature_number)

opeuc_error = np.zeros((num_feature_list_size, sim_rep))
opeuc_cover = np.zeros((num_feature_list_size, sim_rep))

for i, feature_number in enumerate(ratiolearner_feature_number):
    logging.info("Feature: {0} start.".format(feature_number))
    ratiolearner_setting['rbf_ndims'] = feature_number
    wm_ratiolearner_setting['rbf_ndims'] = feature_number

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
        opeuc_baseline_est = opeuc_obj.intercept
        err = opeuc_est - opeuc_truth
        opeuc_error[i, k] = err
        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj.eif_arr)
        opeuc_cover[i, k] = cover_truth(opeuc_ci, opeuc_truth)

        logging.info("COPE: {0}, CI-Cover: {1}".format(opeuc_error[i, k], opeuc_cover[i, k]))
        pass
    pass

np.savetxt('ratio-feature-trl-{0}.csv'.format(state_dim), opeuc_error, delimiter=", ")
np.savetxt('ratio-feature-trl-cover-{0}.csv'.format(state_dim), opeuc_cover, delimiter=", ")
