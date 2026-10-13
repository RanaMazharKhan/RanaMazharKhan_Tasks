# Task 52 - Machine Learning - Logistic Regression
# Problem: define a sigmoid function, print sigmoid(0) and sigmoid(2),
# and say which output is higher.

import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print("sigmoid(0) =", sigmoid(0))   # 0.5
print("sigmoid(2) =", round(float(sigmoid(2)), 3))
# sigmoid(2) is higher - larger input gives higher pass probability.
