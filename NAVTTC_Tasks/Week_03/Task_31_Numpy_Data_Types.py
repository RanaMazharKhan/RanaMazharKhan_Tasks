# Task 31 - NumPy - Data Types
# Problem: Create a float array [2.5, 3.5, 4.5], convert it to int,
# and print both arrays with their dtypes.

import numpy as np
data = np.array([2.5, 3.5, 4.5])
print(data, data.dtype)
print(data.astype(int), data.astype(int).dtype)
