# Task: compare click-through rates of two algorithms with a t-test,
# report the p-value and say if the difference is significant at 0.05.
import numpy as np
from scipy import stats

rng = np.random.default_rng(11)
algo_a = rng.normal(0.045, 0.01, 200)
algo_b = rng.normal(0.052, 0.01, 200)

t_stat, p_value = stats.ttest_ind(algo_a, algo_b)
print("p-value:", p_value)
if p_value < 0.05:
    print("Significant: algorithm B has a different CTR.")
else:
    print("Not significant.")

# Note: significance alone is not enough to switch in production, because the
# actual gain may be too small to be worth the cost and risk of changing.
