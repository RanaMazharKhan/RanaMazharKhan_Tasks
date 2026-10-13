# Task 47 - Pandas - Data Cleaning Techniques
# Problem: Remove the duplicate row from the DataFrame
# {'Name':['Ali','Sara','Ali'],'Age':[20,21,20]} and print the result.

import pandas as pd
df = pd.DataFrame({"Name": ["Ali", "Sara", "Ali"], "Age": [20, 21, 20]})
print(df.drop_duplicates())
