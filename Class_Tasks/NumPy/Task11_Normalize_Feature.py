# Task 11: normalize house_sizes into the range [0, 1] using
# (value - min) / (max - min), in one vectorized line (no loop).
import numpy as np

house_sizes = np.array([850, 900, 1200, 1500, 2000, 750, 1100])
normalized = (house_sizes - house_sizes.min()) / (house_sizes.max() - house_sizes.min())
print(normalized)
