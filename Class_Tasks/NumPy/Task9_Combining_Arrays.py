# Task 9: join group_a and group_b with np.concatenate, and stack them
# as two rows of a 2D array with np.vstack. Print both results.
import numpy as np

group_a = np.array([1, 2, 3])
group_b = np.array([4, 5, 6])

joined = np.concatenate((group_a, group_b))
stacked = np.vstack((group_a, group_b))

print(joined)
print(stacked)
