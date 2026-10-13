# Task: find the correlation between X and Y and draw a scatter plot.
import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 20, 30, 40, 50])
y = np.array([12, 18, 33, 38, 55])

corr = np.corrcoef(x, y)[0, 1]
print("Correlation:", corr)

plt.scatter(x, y)
plt.title("Correlation = " + str(round(corr, 2)))
plt.show()
