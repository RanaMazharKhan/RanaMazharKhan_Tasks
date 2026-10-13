# Task 44 - Pandas - DataFrame
# Problem: Create a DataFrame with columns City and Population for 3
# cities, and print the second row.

import pandas as pd
df = pd.DataFrame({"City": ["Lahore", "Karachi", "Multan"],
                   "Population": [13, 17, 2]})
print(df)
print(df.loc[1])
