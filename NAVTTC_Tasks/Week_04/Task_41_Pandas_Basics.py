# Task 41 - Pandas - Basics / Tutorial Introduction
# Problem: Create a DataFrame of 3 products with columns Product and
# Price, and print it.

import pandas as pd
df = pd.DataFrame({"Product": ["Pen", "Book", "Bag"], "Price": [50, 300, 900]})
print(df)
