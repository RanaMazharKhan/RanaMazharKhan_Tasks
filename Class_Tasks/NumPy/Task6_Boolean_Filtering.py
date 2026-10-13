# Task 6: from scores, extract only the scores >= 70 using boolean
# indexing, then print how many scores passed the filter.
import numpy as np

scores = np.array([55, 89, 76, 92, 40, 67, 98])
passed = scores[scores >= 70]
print(passed)
print("number of scores passed:", len(passed))
