# Task 46 - Pandas - Analyzing DataFrames
# Problem: For a DataFrame of marks [60, 75, 90, 85], print the mean
# marks using describe() or mean().

import pandas as pd
df = pd.DataFrame({"Marks": [60, 75, 90, 85]})
print("Mean marks:", df["Marks"].mean())
