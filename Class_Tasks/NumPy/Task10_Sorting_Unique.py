# Task 10: print the unique grades with their counts, sorted so the
# most common grade comes first.
import numpy as np

grades = np.array(["B", "A", "C", "A", "B", "A", "D"])
unique, counts = np.unique(grades, return_counts=True)
order = np.argsort(-counts)  # sort counts from high to low
for grade, count in zip(unique[order], counts[order]):
    print(grade, ":", count)
