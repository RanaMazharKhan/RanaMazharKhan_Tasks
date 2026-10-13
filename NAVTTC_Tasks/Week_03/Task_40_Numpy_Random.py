# Task 40 - NumPy - Random Numbers
# Problem: Generate 4 random integers between 1 and 50 (use seed 7 so
# the result is fixed) and print them.

from numpy import random
random.seed(7)
print(random.randint(1, 51, size=4))
