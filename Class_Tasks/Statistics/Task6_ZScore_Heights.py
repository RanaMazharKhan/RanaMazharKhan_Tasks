# Task: turn heights into z-scores and check the mean is ~0 and std is ~1.
import numpy as np

heights = np.array([150, 160, 165, 170, 180, 190])
z = (heights - heights.mean()) / heights.std(ddof=1)
print("Z-scores:", z)
print("Mean of z:", z.mean())      # about 0
print("Std of z:", z.std(ddof=1))  # about 1
