# Task: standardize two features with very different scales and check
# both have the same std dev (1) after standardization.
import numpy as np

a = np.random.default_rng(3).normal(50, 5, 200)
b = np.random.default_rng(3).normal(500, 300, 200)

za = (a - a.mean()) / a.std(ddof=1)
zb = (b - b.mean()) / b.std(ddof=1)

print("Before:", a.std(ddof=1), b.std(ddof=1))
print("After:", za.std(ddof=1), zb.std(ddof=1))
