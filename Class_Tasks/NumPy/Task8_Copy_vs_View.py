# Task 8: make a view of original[1:3], change one element and show
# original changed too. Then do the same with .copy() and show
# original is unaffected.
import numpy as np

original = np.array([5, 10, 15, 20, 25])
partial_view = original[1:3]
partial_view[0] = 99
print("after changing the view, original:", original)

original = np.array([5, 10, 15, 20, 25])
safe_copy = original[1:3].copy()
safe_copy[0] = 99
print("after changing the copy, original:", original)
