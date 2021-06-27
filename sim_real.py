from sklearn.linear_model import LogisticRegression, MultiTaskElasticNet
import numpy as np
from opeuc import opeuc_run
from simulator_real import Simulator
from utilize_ci import compute_ci, cover_truth

state_dim = 3
gamma = 0.9

s0 = [np.array([2.254948, 2.232579]),
      np.array([[1.60432501, 0.04516279], [0.04516279, 1.10574671]])]
# print(s0)

y = np.array([0, 1, 0, 1])
x_a = np.array([[1], [2], [3], [4]])
pa = LogisticRegression()
pa.fit(x_a, y)
pa.coef_ = np.array([[0.01291389, 0.03131658]])
pa.intercept_ = np.array([0.07991098])
# print(pa.coef_)

x_m = np.array([[1, 2], [2, 1], [1, 0], [0, 1]])
pm = LogisticRegression()
pm.fit(x_m, y)
pm.coef_ = np.array([[1.85704982e-03, -4.24983441e-04, 2.23051766]])
pm.intercept_ = np.array([-1.08132472])
# print(pm.coef_)

x_r = np.array([[1, 2, 3], [2, 1, 3], [1, 0, -1], [0, 1, -3]])
pr_model = MultiTaskElasticNet(
    alpha=0.03125, copy_X=False, l1_ratio=0.5, random_state=1, tol=1e-06, warm_start=True)
pr_model.fit(x_r, y.reshape(-1, 1))
pr_model.coef_ = np.array([[0.01026785, 0.43406879, 0.01970068, -0.22547776]])
pr_model.intercept_ = np.array([1.44034209])
pr = {'estimator': pr_model,
      'noise': np.array([0.10909449]), 
      'params': {'menet__alpha': 0.03125, 'menet__l1_ratio': 0.05, 'poly_features__degree': 1, 'poly_features__interaction_only': False}}
# print(pa.coef_)

y_ns = np.array([[0, 1], [0, 1], [0, 1], [0, 1]])
pns_model = MultiTaskElasticNet(alpha=0.03125, copy_X=False, l1_ratio=0.5, random_state=1, tol=1e-06, warm_start=True)
pns_model.fit(x_r, y_ns)
pns_model.coef_ = np.array([[2.33946860e-01, -2.97368785e-02, 0.0, 0.0,
                             1.70187886e-01, 6.32233240e-02, 1.06307011e-02, 4.67770700e-03,
                             -1.53111710e-02, 0.0, -1.31049989e-04, 0.0,
                             0.0, 0.0, -1.22751522e-02, -9.81441308e-03,
                             -2.72928565e-03, -6.18174396e-04, -2.73935402e-03, -5.03704770e-03,
                             -3.44805089e-03, 1.06946955e-02, 0.0, 4.73783386e-03, 
                             3.36757445e-03, 1.99222100e-03, 1.24778909e-03, 0.0,
                             4.02733589e-04, -1.29521552e-04, 0.0, 0.0,
                             0.0, 0.0],
                            [1.75750507e-02, 8.18547680e-02, 0.0, 0.0,
                             5.19404911e-03, 2.72571406e-02, -5.16478631e-03, 1.10764759e-03,
                             1.69809829e-01, 0.0, 5.26738205e-03, 0.0,
                             0.0, 0.0, -9.27687993e-04, -4.82379465e-03,
                             9.67735882e-04, -2.37310645e-05, -1.93238454e-03, -5.07797639e-04,
                             -4.38594047e-04, -5.19827385e-03, 0.0, 1.11736426e-03, 
                             -2.21702965e-02, 8.81602842e-04, -4.52276066e-03, 0.0,
                             1.00623185e-03, 5.27881519e-03, 0.0, 0.0,
                             0.0, 0.0]])
pns_model.intercept_ = np.array([1.08287167, 1.36688502])
pns = {'estimator': pns_model,
       'noise': np.array([[0.23744044, 0.0107695], [0.0107695, 0.8846272]]),
       'params': {'poly_features__degree': 3, 'poly_features__interaction_only': False,
                  'menet__alpha': 0.03125, 'menet__l1_ratio': 0.05}}

pt_model = MultiTaskElasticNet(alpha=0.03125, copy_X=False, l1_ratio=0.5, random_state=1, tol=1e-06, warm_start=True)
pt_model.fit(x_r, y.reshape(-1, 1))
pt_model.coef_ = np.array([0.07139351, 0.0, 0.01910431, 0.02252674, -0.02847763, -0.00458277,
                           -0.00763392, 0.0, 0.02317713, 0.03555253, -0.01701916, 0.01910338,
                           0.03579361, 0.02252673])
pt_model.intercept_ = np.array([3.14743637])
pt = {'estimator': pt_model,
      'params': {'poly_features__degree': 2, 'poly_features__interaction_only': False,
                 'menet__alpha': 0.03125, 'menet__l1_ratio': 0.05},
      'noise': np.array([13.40315297])}


def didi_baseline_policy2(state, action=None):
    pa = 0.5
    pa = np.array([1-pa, pa])
    if action is None:
        action_value = np.random.choice([0.0, 1.0], 1, p=pa)
    else:
        action_value = np.array(pa[int(action)])
        pass
    return action_value


def didi_improve_policy2(state, action=None):
    if state[0] <= 3.5:
        pa = 0.75
    else:
        pa = 0.25

    pa = np.array([1 - pa, pa])
    if action is None:
        action_value = np.random.choice([0.0, 1.0], 1, p=pa)
    else:
        action_value = np.array([pa[int(action)]])
        pass
    return action_value


simulator = Simulator(s0, pa, pm, pr, pns, pt)
# ope_truth_baseline = simulator.estimate_ope(didi_baseline_policy2, 0.9, max_time=500, mc_s0_time=100, timegap=True)
# print(ope_truth_baseline)
# ope_truth_improved = simulator.estimate_ope(didi_improve_policy2, 0.9, max_time=500, mc_s0_time=100, timegap=True)
# print(ope_truth_improved)

# 50000 trajectories * 5000 times: 
ope_truth_baseline = 8.075078894939093
ope_truth_improved = 8.127298546443104

truth_gap = ope_truth_improved - ope_truth_baseline
print(truth_gap)

num_trajectory_list = [320]
num_time = 20
sim_rep = 100

seed_list = list(range(sim_rep))
num_trajectory_list_size = len(num_trajectory_list)

cope_error = np.zeros((num_trajectory_list_size, sim_rep))
is_error = np.zeros((num_trajectory_list_size, sim_rep))
direct_error = np.zeros((num_trajectory_list_size, sim_rep))
cope_cover = np.zeros((num_trajectory_list_size, sim_rep))
is_cover = np.zeros((num_trajectory_list_size, sim_rep))
direct_cover = np.zeros((num_trajectory_list_size, sim_rep))

palearner_setting = {'discrete_state': False, 'rbf_dim': 1, 'cv_score': 'accuracy', 'verbose': True}
pmlearner_setting = {'discrete_state': False, 'discrete_action': False, 'rbf_dim': 1, 'cv_score': 'accuracy', 'verbose': True}
qlearner_setting = {'epoch': 500, 'trace': False, 'rbf_dim': 100, 'verbose': True, 'model': 'linear', 'eps': 1e-8}
ratiolearner_setting = {'mode': 'linear', 'rbf_ndims': 100, 'batch_size': 32, 'epoch': 3, 'lr': 0.01, 'verbose': True}

##################################################################
##################### Trajectory comparison ######################
##################################################################
for i, num_trajectory in enumerate(num_trajectory_list):
    # logging.info("Trajectory: {0} start.".format(num_trajectory))
    for k, r in enumerate(seed_list):
        if r % 10 == 0:
            print("Remain: ", 100 * (sim_rep - r) / sim_rep, "%.")
        pass
        if r != 0:
            print("Error: ", np.mean(direct_error[i, range(r)]), np.mean(is_error[i, range(r)]), np.mean(cope_error[i, range(r)]))
            print("Rate: ", np.mean(direct_cover[i, range(r)]), np.mean(is_cover[i, range(r)]), np.mean(cope_cover[i, range(r)]))
            pass

        simulator.sample_trajectory(num_trajectory, num_time, seed=r)
        simulator.trajectory2iid()
        sim_iid_dataset = simulator.iid_dataset
        s0 = sim_iid_dataset['s0']
        time_difference = np.copy(sim_iid_dataset['timegap'])
        iid_dataset = []
        iid_dataset.append(sim_iid_dataset['state'])
        iid_dataset.append(sim_iid_dataset['action'])
        iid_dataset.append(sim_iid_dataset['mediator'])
        iid_dataset.append(sim_iid_dataset['reward'])
        iid_dataset.append(sim_iid_dataset['next_state'])
        
        ## cope:
        opeuc_obj1 = opeuc_run(s0, iid_dataset, didi_baseline_policy2, time_difference, gamma=gamma, 
                                palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting, 
                                qlearner_setting=qlearner_setting, ratiolearner_setting=ratiolearner_setting)
        opeuc_obj2 = opeuc_run(s0, iid_dataset, didi_improve_policy2, time_difference, gamma=gamma, 
                                palearner_setting=palearner_setting, pmlearner_setting=pmlearner_setting, 
                                qlearner_setting=qlearner_setting, ratiolearner_setting=ratiolearner_setting)
        cope_est = opeuc_obj2.opeuc - opeuc_obj1.opeuc
        is_est = opeuc_obj2.cis - opeuc_obj1.cis
        direct_est = opeuc_obj2.intercept - opeuc_obj1.intercept
        cope_error[i, k] = cope_est - truth_gap
        is_error[i, k] = is_est - truth_gap
        direct_error[i, k] = direct_est - truth_gap

        opeuc_ci = compute_ci(num_trajectory, num_time, opeuc_obj2.eif_arr - opeuc_obj1.eif_arr)
        cope_cover[i, k] = cover_truth(opeuc_ci, truth_gap)
        is_ci = compute_ci(num_trajectory, num_time, opeuc_obj2.cis_arr - opeuc_obj1.cis_arr)
        is_cover[i, k] = cover_truth(is_ci, truth_gap)
        direct_ci = compute_ci(num_trajectory, 1, opeuc_obj2.intercept_arr - opeuc_obj1.intercept_arr)
        direct_cover[i, k] = cover_truth(direct_ci, truth_gap)
        pass
    pass

