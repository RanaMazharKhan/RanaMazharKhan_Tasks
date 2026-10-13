# Task 38 - NumPy - Array Searching
# Problem: In the array [3, 6, 9, 6, 12], print the indexes where the
# value is 6.

import numpy as np
arr = np.array([3, 6, 9, 6, 12])
print(np.where(arr == 6)[0])
