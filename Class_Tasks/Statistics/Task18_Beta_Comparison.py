# Task: plot Beta(2,2) vs Beta(10,10) and say which is more concentrated at 0.5.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 1, 300)
plt.plot(x, stats.beta(2, 2).pdf(x), label="Beta(2,2)")
plt.plot(x, stats.beta(10, 10).pdf(x), label="Beta(10,10)")
plt.legend()
plt.title("Beta(2,2) vs Beta(10,10)")
plt.show()

print("Beta(10,10) is more concentrated around 0.5.")
