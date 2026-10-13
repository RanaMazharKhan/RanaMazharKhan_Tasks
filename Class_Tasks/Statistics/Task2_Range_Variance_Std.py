# Task: for the same values, find the range, sample variance and sample std dev.
import numpy as np

values = [12, 15, 12, 18, 20, 12, 25]
print("Range:", max(values) - min(values))
print("Sample variance:", np.var(values, ddof=1))
print("Sample std dev:", np.std(values, ddof=1))
