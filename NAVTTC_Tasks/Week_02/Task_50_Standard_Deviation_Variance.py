# Task 50 - Statistics - Standard Deviation and Variance
# Problem: Heights in cm: [160, 165, 170, 165, 180]. Print the
# variance and standard deviation.

import numpy as np
heights = [160, 165, 170, 165, 180]
print("Variance:", np.var(heights))
print("Std deviation:", round(float(np.std(heights)), 2))
