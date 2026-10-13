# Day 9 summary question 1: What is ML?
# Machine learning is when the computer learns a pattern from data
# instead of us writing fixed rules. Here it learns price from area by itself.

import os
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

small = df[["area", "price"]].dropna()
model = LinearRegression()
model.fit(small[["area"]], small["price"])  # no rule written, it learns from data
print("Learned rule: price =", round(model.coef_[0], 1), "* area +", round(model.intercept_, 1))
