# Task 5: convert celsius = [0, 10, 20, 30, 40] to Fahrenheit using
# F = C * 9/5 + 32, in one vectorized line (no loop).
import numpy as np

celsius = np.array([0, 10, 20, 30, 40])
fahrenheit = celsius * 9 / 5 + 32
print(fahrenheit)
