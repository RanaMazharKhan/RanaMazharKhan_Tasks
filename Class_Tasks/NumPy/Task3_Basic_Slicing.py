# Task 3: create values = np.arange(10, 60, 5). Print the first 3
# elements, the last 3 elements, and every second element.
import numpy as np

values = np.arange(10, 60, 5)
print(values)
print("first 3:", values[:3])
print("last 3:", values[-3:])
print("every second:", values[::2])
