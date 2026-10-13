# Task 49 - Statistics - Mean, Median and Mode
# Problem: For the data [10, 20, 20, 30, 40], find and print the mean,
# median and mode.

import numpy as np
from scipy import stats
data = [10, 20, 20, 30, 40]
print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Mode:", stats.mode(data, keepdims=True).mode[0])
