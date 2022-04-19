import numpy as np
from opeuc import opeuc_run
from simulator_save import Simulator
from policy import target_policy
from utilize_ci import compute_ci, cover_truth

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='ratio_nn.log')

state_dim = 1
simulator = Simulator(dim_state=state_dim, model_type='save')

palearner_setting={'discrete_state': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting={'discrete_state': False, 'discrete_action': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
qlearner_setting = {'epoch': 100, 'trace': False, 'rbf_dim': [5], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
# ratiolearner_setting={'mode':'ANN', 'rbf_ndims': [5], 'batch_size':32, 'epoch':3, 'lr':0.01, 'verbose': True}
ratiolearner_setting={'mode':'linear', 'rbf_ndims': [5], 'batch_size':32, 'epoch':3, 'lr':0.01, 'verbose': True}
opeuc_truth = 1.4498987620607517

gamma = 0.9
sim_rep = 50

num_trajectory_list = [20, 40, 80, 160, 320]
num_time = 20
seed_list = list(range(sim_rep))
num_trajectory_list_size = len(num_trajectory_list)

opeuc_error = np.zeros((num_trajectory_list_size, sim_rep))
opeuc_cover = np.zeros((num_trajectory_list_size, sim_rep))

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
        opeuc_baseline_est = opeuc_obj.intercept
        err = opeuc_est - opeuc_truth
        opeuc_error[i, k] = err
        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj.eif_arr)
        opeuc_cover[i, k] = cover_truth(opeuc_ci, opeuc_truth)

        logging.info("COPE: {0}, CI-Cover: {1}".format(opeuc_error[i, k], opeuc_cover[i, k]))
        pass
    pass

if ratiolearner_setting['mode'] == 'ANN':
    np.savetxt('ratio-nn-feature-trl-{0}.csv'.format(state_dim), opeuc_error, delimiter=", ")
    np.savetxt('ratio-nn-feature-trl-cover-{0}.csv'.format(state_dim), opeuc_cover, delimiter=", ")
else:
    np.savetxt('ratio-linear-feature-trl-{0}.csv'.format(state_dim), opeuc_error, delimiter=", ")
    np.savetxt('ratio-linear-feature-trl-cover-{0}.csv'.format(state_dim), opeuc_cover, delimiter=", ")
