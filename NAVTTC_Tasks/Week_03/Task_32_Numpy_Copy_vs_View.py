# Task 32 - NumPy - Array Copy vs View
# Problem: Make an array [5,6,7]. Make a copy, change the copy's first
# element to 100, and print the original to prove it did not change.

import numpy as np
arr = np.array([5, 6, 7])
new_arr = arr.copy()
new_arr[0] = 100
print("Original (unchanged):", arr)
print("Copy:", new_arr)
