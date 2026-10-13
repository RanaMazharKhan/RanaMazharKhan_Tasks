# Task 36 - NumPy - Array Joining
# Problem: Join the arrays [10, 20] and [30, 40, 50] into one array
# and print it.

import numpy as np
joined = np.concatenate((np.array([10, 20]), np.array([30, 40, 50])))
print(joined)
