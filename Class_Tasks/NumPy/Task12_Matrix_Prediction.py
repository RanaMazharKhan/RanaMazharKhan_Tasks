# Task 12: compute predictions for all 3 samples in one line using @,
# then find the index of the sample with the highest prediction.
import numpy as np

X = np.array([
    [2, 3, 1],
    [5, 1, 4],
    [1, 1, 1],
])
weights = np.array([0.4, 0.4, 0.2])

predictions = X @ weights
print(predictions)
print("best sample index:", np.argmax(predictions))
