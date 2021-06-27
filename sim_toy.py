import numpy as np
from scipy.special import expit
from opeuc import opeuc_run
from opedr import opedr_run
from simulator_save import Simulator
from policy import target_policy
from utilize_ci import compute_ci, cover_truth

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='sim_toy.log')

gamma = 0.9
state_dim = 1
simulator = Simulator(dim_state=state_dim, model_type='toy')

palearner_setting = {'discrete_state': False, 'rbf_dim': 1, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting = {'discrete_state': False, 'discrete_action': False, 'rbf_dim': 2, 'cv_score': 'accuracy', 'verbose': True}
qlearner_setting={'epoch': 100, 'trace': False, 'rbf_dim': [20], 'verbose': True, 'model': 'linear', 'eps': 1e-5}
ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': [1], 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}

truth = 55.0559
num_time = 200

nrep = 500
# num_trajectory_list = np.arange(1, 16) * 60
num_trajectory_list = np.arange(1, 3) * 60
num_trajectory_list_size = num_trajectory_list.shape[0]

error0 = np.zeros((num_trajectory_list_size, nrep))
cover0 = np.zeros((num_trajectory_list_size, nrep))

for i, num_trajectory in enumerate(num_trajectory_list):
    print("Trajectory:", num_trajectory, "start.")
    for r in range(nrep):
        if r % 10 == 0:
            print("Remain: ", (nrep - r) / nrep, "%.")
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

        opeuc_obj0 = opeuc_run(s0, iid_dataset, target_policy, palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting,
                               qlearner_setting=qlearner_setting, ratiolearner_setting=ratiolearner_setting)
        trl_value0 = opeuc_obj0.opeuc
        error0[i, r] = trl_value0 - truth
        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj0.eif_arr)
        cover0[i, r] = cover_truth(opeuc_ci, truth)

        logging.info("COPE0: {0}, CI-Cover: {1}".format(error0[i, r], cover0[i, r]))
        pass
    print("Trajectory: ", num_trajectory, "done.")
    pass

np.savetxt('toy-cope.csv', error0, delimiter=',')
np.savetxt('toy-cope-cover.csv', cover0, delimiter=',')
