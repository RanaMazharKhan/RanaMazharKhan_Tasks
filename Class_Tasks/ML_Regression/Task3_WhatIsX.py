# Day 9 summary question 3: What is X?
# X is the features table - all the columns the model is allowed to look at
# to make its prediction. Everything except the target (price).

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

X = df.drop(columns=["price"])
print("X = features, shape:", X.shape)
print("Columns in X:", list(X.columns))
