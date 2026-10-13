# Task: run a two-sample t-test on normal(60, 10, 40) vs normal(65, 10, 40)
# and give the conclusion at alpha = 0.05.
import numpy as np
from scipy import stats

rng = np.random.default_rng(9)
group1 = rng.normal(60, 10, 40)
group2 = rng.normal(65, 10, 40)

t_stat, p_value = stats.ttest_ind(group1, group2)
print("p-value:", p_value)
if p_value < 0.05:
    print("Reject H0: the means are significantly different.")
else:
    print("Fail to reject H0: no significant difference.")
