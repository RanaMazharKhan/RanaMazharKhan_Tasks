# Task 43 - Pandas - Series Data
# Problem: Create a Series of ages with custom labels ['s1','s2','s3']
# for values [20, 21, 19] and print the age of 's2'.

import pandas as pd
ages = pd.Series([20, 21, 19], index=["s1", "s2", "s3"])
print(ages)
print("Age of s2:", ages["s2"])
