# Task 45 - Pandas - Read / Open CSV Files
# Problem: Read CSV text with columns Student,Score for 3 students and
# print only the Score column.

import pandas as pd, io
csv_text = "Student,Score\nHina,91\nBilal,84\nZara,77\n"
df = pd.read_csv(io.StringIO(csv_text))
print(df["Score"])
