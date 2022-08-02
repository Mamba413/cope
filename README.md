
# Off-Policy Interval Estimation with Confounded Markov Decision Process (COPE)

This repository contains the implementation for the paper "Off-Policy Interval Estimation with Confounded Markov Decision Process" (JASA, 2022+) in Python.

## Requirements
Change your working directory to this main folder, run `conda env create --file COPE.yml` to create the Conda environment, 
and then run `conda activate COPE` to activate the environment. 

## Code Description

The proposed estimators:
- `opeuc.py`: direct estimator, importance sampling estimator, confounded off-policy estimator

Nuisance parameters:
- `problearner.py`: learn transition probabilities: (i) state --> action & (ii) (state, action) --> mediator
- `qlearner.py`: fitted Q evaluation
- `rnnl.py`: marginal ratio learning via neural network
- `rll.py`: marginal ratio learning via linear model

Sampling:
- `simulator_save.py`: generate observations tuple from MDP

Utilities:
- `policy.py`: target policies
- `utilize.py`: some helpful functions
- `utilize_ci.py`: helpful functions for computing confidence intervals
- `utilize_prototypemodel.py`: helpful classes for simulations

Numerical experiments:

(See ACC form detailed for instructions)

- `sim_robust.py`: simulation for demonstrating double robustness
- `sim_time_compare.py` & `sim_time_compare_multdim.py`: simulation when time points vary
- `sim_trajectory_compare.py` & `sim_trajectory_compare_multdim.py`: simulation when the number of trajectories vary
- `sim_ratiolearner_compare.py`
- `sim_ratio_features_number_compare.py`

## Citations

Please cite the following publications if you make use of the material here. 

- Chengchun Shi, Jin Zhu, Ye Shen, Shikai Luo, Hongtu Zhu, Rui Song (2022). Off-Policy Confidence Interval Estimation with Confounded Markov Decision Process. aarXiv preprint arXiv:2202.10589, 2022.

```
@article{zhang2021certifiably,
  title = {Off-Policy Confidence Interval Estimation with Confounded Markov Decision Process},
  author = {Chengchun Shi, Jin Zhu, Ye Shen, Shikai Luo, Hongtu Zhu, Rui Song},
  journal = {arXiv preprint arXiv:2202.10589},
  year = {2022}
}
```

## License

All content in this repository is licensed under the GPL-3 license.
