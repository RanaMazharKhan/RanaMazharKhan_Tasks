# Task 2: create a 4x4 array of zeros, a 3x3 array filled with 9,
# and a 5x5 identity matrix. Print all three.
import numpy as np

zeros = np.zeros((4, 4))
nines = np.full((3, 3), 9)
identity = np.eye(5)

print(zeros)
print(nines)
print(identity)
