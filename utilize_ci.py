import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_std(num, time, data_point_estimation):
    trajectory_id = np.repeat(np.arange(num), time)
    df = pd.DataFrame(data={'id': trajectory_id, 'trajectory_estimation': data_point_estimation})
    df = df.groupby('id').mean()
    std_value = df['trajectory_estimation'].to_numpy().std()
    return std_value

def compute_ci(num, time, data_point_estimation, alpha=0.95):
    std_value = compute_std(num, time, data_point_estimation)
    se_value = std_value / np.sqrt(np.array(num*1.0))
    est = np.mean(data_point_estimation)
    norm_alpha = 1 - (1 - alpha) / 2
    quan_alpha = norm.ppf(norm_alpha)
    ci = np.array([est - quan_alpha*se_value, est + quan_alpha*se_value])
    return ci

def cover_truth(ci, truth):
    cover = truth >= ci[0] and truth <= ci[1]
    cover = 1.0 * cover
    return cover

def compute_diff_ci(num, time, data_point_estimation1, data_point_estimation2, alpha=0.95):
    std_value1 = compute_std(num, time, data_point_estimation1)
    std_value2 = compute_std(num, time, data_point_estimation2)
    se_value = np.sqrt(np.square(std_value1) + np.square(std_value2)) / np.sqrt(np.array(num*1.0))
    est = np.mean(data_point_estimation2 - data_point_estimation1)
    norm_alpha = 1 - (1 - alpha) / 2
    quan_alpha = norm.ppf(norm_alpha)
    ci = np.array([est - quan_alpha*se_value, est + quan_alpha*se_value])
    return ci
