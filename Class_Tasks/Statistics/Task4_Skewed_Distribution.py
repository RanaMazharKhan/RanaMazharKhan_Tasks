# Task: make 1000 samples from an exponential (right-skewed) distribution,
# find mean, median and skewness, and plot a histogram with mean and median lines.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sample = np.random.default_rng(2).exponential(scale=2.0, size=1000)
print("Mean:", sample.mean())
print("Median:", np.median(sample))
print("Skewness:", stats.skew(sample))

plt.hist(sample, bins=40)
plt.axvline(sample.mean(), color="red", label="Mean")
plt.axvline(np.median(sample), color="orange", label="Median")
plt.legend()
plt.title("Right-skewed sample")
plt.show()
