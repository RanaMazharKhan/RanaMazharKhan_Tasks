# Day 9 summary question 4: What is y?
# y is the target - the single column we want to predict. Here y is price.

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

y = df["price"]
print("y = target (price)")
print("First 5 values:", list(y.head()))
print("Cheapest:", y.min(), "| Most expensive:", y.max())
