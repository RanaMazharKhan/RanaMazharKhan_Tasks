# Task: for values = [12, 15, 12, 18, 20, 12, 25], find the mean, median and mode.
import numpy as np

values = [12, 15, 12, 18, 20, 12, 25]
print("Mean:", np.mean(values))
print("Median:", np.median(values))
print("Mode:", max(set(values), key=values.count))
