# NumPy practice tasks

import numpy as np

# ---

# Task 1: create a 1D array "temperatures" from [30, 32, 29, 35, 31]
# and print its shape, ndim, size and dtype. Also create a 4x4 array
# of zeros, a 3x3 array of 9s and a 5x5 identity matrix.
temperatures = np.array([30, 32, 29, 35, 31])
print(temperatures)
print("shape:", temperatures.shape)
print("ndim:", temperatures.ndim)
print("size:", temperatures.size)
print("dtype:", temperatures.dtype)

zeros = np.zeros((4, 4))
nines = np.full((3, 3), 9)
identity = np.eye(5)

print(zeros)
print(nines)
print(identity)

# ---

# Task 2: create values = np.arange(10, 60, 5). Print the first 3
# elements, the last 3 elements, and every second element.
values = np.arange(10, 60, 5)
print(values)
print("first 3:", values[:3])
print("last 3:", values[-3:])
print("every second:", values[::2])

# ---

# Task 3: from the 2D grid below print (a) the element at row 2,
# column 1, (b) the entire second row, (c) the entire first column.
grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print("element at row 2, column 1:", grid[2, 1])
print("second row:", grid[1])
print("first column:", grid[:, 0])

# ---

# Task 4: convert celsius = [0, 10, 20, 30, 40] to Fahrenheit using
# F = C * 9/5 + 32, in one vectorized line (no loop).
celsius = np.array([0, 10, 20, 30, 40])
fahrenheit = celsius * 9 / 5 + 32
print(fahrenheit)

# ---

# Task 5: from scores, extract only the scores >= 70 using boolean
# indexing, then print how many scores passed the filter.
scores = np.array([55, 89, 76, 92, 40, 67, 98])
passed = scores[scores >= 70]
print(passed)
print("number of scores passed:", len(passed))

# ---

# Task 6: create numbers = np.arange(1, 25), reshape into (4, 6).
# Print the sum of each row (axis=1) and each column (axis=0).
numbers = np.arange(1, 25).reshape(4, 6)
print(numbers)
print("row sums:", numbers.sum(axis=1))
print("column sums:", numbers.sum(axis=0))

# ---

# Task 7: make a view of original[1:3], change one element and show
# original changed too. Then do the same with .copy() and show
# original is unaffected.
original = np.array([5, 10, 15, 20, 25])
partial_view = original[1:3]
partial_view[0] = 99
print("after changing the view, original:", original)

original = np.array([5, 10, 15, 20, 25])
safe_copy = original[1:3].copy()
safe_copy[0] = 99
print("after changing the copy, original:", original)

# ---

# Task 8: join group_a and group_b with np.concatenate, and stack them
# as two rows of a 2D array with np.vstack. Print both results.
group_a = np.array([1, 2, 3])
group_b = np.array([4, 5, 6])

joined = np.concatenate((group_a, group_b))
stacked = np.vstack((group_a, group_b))

print(joined)
print(stacked)

# ---

# Task 9: print the unique grades with their counts, sorted so the
# most common grade comes first.
grades = np.array(["B", "A", "C", "A", "B", "A", "D"])
unique, counts = np.unique(grades, return_counts=True)
order = np.argsort(-counts)  # sort counts from high to low
for grade, count in zip(unique[order], counts[order]):
    print(grade, ":", count)

# ---

# Task 10: normalize house_sizes into the range [0, 1] using
# (value - min) / (max - min), in one vectorized line (no loop).
house_sizes = np.array([850, 900, 1200, 1500, 2000, 750, 1100])
normalized = (house_sizes - house_sizes.min()) / (house_sizes.max() - house_sizes.min())
print(normalized)

# ---

# Task 11: compute predictions for all 3 samples in one line using @,
# then find the index of the sample with the highest prediction.
X = np.array([
    [2, 3, 1],
    [5, 1, 4],
    [1, 1, 1],
])
weights = np.array([0.4, 0.4, 0.2])

predictions = X @ weights
print(predictions)
print("best sample index:", np.argmax(predictions))

# ---

# Task 12: with only vectorized numpy (no loops): 1) each student's
# average (axis=1), 2) each subject's class average (axis=0), 3) students
# with average >= 80 using boolean indexing, 4) save the per-student
# averages to a temp .npy file, load it back and print it.
student_marks = np.array([
    [78, 85, 90],
    [88, 92, 95],
    [55, 62, 58],
    [90, 87, 93],
])

student_averages = student_marks.mean(axis=1)
subject_averages = student_marks.mean(axis=0)

print("per-student averages:", student_averages)
print("per-subject averages:", subject_averages)
print("students with average >= 80:", student_averages[student_averages >= 80])

np.save("/tmp/student_averages.npy", student_averages)
loaded = np.load("/tmp/student_averages.npy")
print("loaded back:", loaded)
