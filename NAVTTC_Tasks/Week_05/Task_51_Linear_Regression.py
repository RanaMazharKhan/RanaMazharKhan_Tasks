# Task 51 - Machine Learning - Linear Regression
# Problem: Using the data x = [1,2,3,4] and y = [10,20,30,40] (perfect
# line y = 10x), train the gradient-descent code pattern above (or the
# closed-form formula) and print the predicted y for x = 5. Expected:
# about 50.

import numpy as np
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([10, 20, 30, 40], dtype=float)
slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
intercept = y.mean() - slope * x.mean()
print("Slope:", slope, "Intercept:", intercept)
print("Prediction for x=5:", slope * 5 + intercept)
