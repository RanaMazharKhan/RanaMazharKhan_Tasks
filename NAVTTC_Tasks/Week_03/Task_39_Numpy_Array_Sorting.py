# Task 39 - NumPy - Array Sorting
# Problem: Sort the array [45, 12, 78, 34, 9] and print the smallest
# and largest value from the sorted result.

import numpy as np
arr = np.sort(np.array([45, 12, 78, 34, 9]))
print("Sorted:", arr)
print("Smallest:", arr[0], "Largest:", arr[-1])
