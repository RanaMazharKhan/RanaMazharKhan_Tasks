# Day 9 summary question 5: What is a feature? Give one example.
# A feature is one input column the model uses. Example: area -
# bigger area usually means higher price, which the model can learn.

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

print("Feature example: area (size of the property)")
print(df[["area", "price"]].dropna().head())
print("Correlation of area with price:", round(df[["area", "price"]].dropna().corr().iloc[0, 1], 2))
