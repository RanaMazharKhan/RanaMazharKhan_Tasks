# Task 35 - NumPy - Array Iterating
# Problem: Use a loop to print each element of the array [7, 14, 21,
# 28] multiplied by 2.

import numpy as np
arr = np.array([7, 14, 21, 28])
for value in arr:
    print(value * 2, end=" ")
print()
