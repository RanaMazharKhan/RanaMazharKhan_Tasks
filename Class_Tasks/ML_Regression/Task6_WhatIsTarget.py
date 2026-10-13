# Day 9 summary question 6: What is a target?
# The target is the answer column we train the model to predict.
# In our project the target is price.

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

target = "price"
print("Target column:", target)
print("It is the answer we want the model to guess for new listings.")
print("Average price:", round(df[target].mean()))
