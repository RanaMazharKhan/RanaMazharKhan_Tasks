# Task 37 - NumPy - Array Splitting
# Problem: Split the array [10,20,30,40,50,60,70,80] into 4 equal
# parts and print the second part.

import numpy as np
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80])
parts = np.array_split(arr, 4)
print("Second part:", parts[1])
