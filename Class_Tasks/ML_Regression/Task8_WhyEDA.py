# Day 9 summary question 8: Why do EDA before modeling?
# Because the data is never clean. In our dataset we found missing values
# in agent, baths and area - a model cannot train on empty cells, so we
# must fill or drop them before training.

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print("Reason for EDA: we found 8 missing baths and 8 missing area values,")
print("so we fill them with the median before training the model.")
