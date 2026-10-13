# Task: take 3000 sample means from an exponential distribution with sample
# size 2 and again with 100, plot both and compare their shapes.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
means_2 = [rng.exponential(scale=3.0, size=2).mean() for _ in range(3000)]
means_100 = [rng.exponential(scale=3.0, size=100).mean() for _ in range(3000)]

plt.subplot(1, 2, 1)
plt.hist(means_2, bins=40)
plt.title("Sample size 2")

plt.subplot(1, 2, 2)
plt.hist(means_100, bins=40)
plt.title("Sample size 100")

plt.show()
# The bigger sample gives a bell-shaped (normal) histogram: this is the CLT.
