# Task 4: from the 2D grid below print (a) the element at row 2,
# column 1, (b) the entire second row, (c) the entire first column.
import numpy as np

grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print("element at row 2, column 1:", grid[2, 1])
print("second row:", grid[1])
print("first column:", grid[:, 0])
