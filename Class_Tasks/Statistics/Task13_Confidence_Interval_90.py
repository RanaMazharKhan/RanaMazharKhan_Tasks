# Task: build a 90% confidence interval for a sample of 35 from normal(80, 15).
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)
sample = rng.normal(loc=80, scale=15, size=35)

low, high = stats.t.interval(0.90, len(sample) - 1,
                             loc=sample.mean(),
                             scale=stats.sem(sample))
print("90% CI:", low, "to", high)
print("Width:", high - low)
