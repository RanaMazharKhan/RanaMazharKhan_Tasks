# Day 9 summary question 2: What problem are we solving?
# We are predicting the price of a property from its other details
# (area, baths, location, ...). Price is a number, so this is regression.

import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

print("Problem: predict price of a property")
print("Rows:", len(df), "| price is a continuous number -> regression problem")
print("Example row:")
print(df[["area", "baths", "location", "price"]].head(1).to_string(index=False))
