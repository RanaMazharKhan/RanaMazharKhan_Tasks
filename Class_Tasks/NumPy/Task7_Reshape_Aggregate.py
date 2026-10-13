# Task 7: create numbers = np.arange(1, 25), reshape into (4, 6).
# Print the sum of each row (axis=1) and each column (axis=0).
import numpy as np

numbers = np.arange(1, 25).reshape(4, 6)
print(numbers)
print("row sums:", numbers.sum(axis=1))
print("column sums:", numbers.sum(axis=0))
