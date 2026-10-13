# Task: find the 25th, 50th and 75th percentiles of values.
import numpy as np

values = [12, 15, 12, 18, 20, 12, 25]
for p in [25, 50, 75]:
    print(p, "th percentile:", np.percentile(values, p))
