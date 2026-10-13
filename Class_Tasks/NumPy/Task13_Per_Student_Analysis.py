# Task 13: with only vectorized numpy (no loops): 1) each student's
# average (axis=1), 2) each subject's class average (axis=0), 3) students
# with average >= 80 using boolean indexing, 4) save the per-student
# averages to student_averages.npy, load it back and print it.
import numpy as np

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

np.save("student_averages.npy", student_averages)
loaded = np.load("student_averages.npy")
print("loaded back:", loaded)
