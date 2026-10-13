# Task: preview a random sample of 5 rows to sanity-check the new column.

import os
import re
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})
df = df.drop_duplicates()
df = df.drop(columns=["agent", "name"])
df = df[df["purpose"].isin(["For Rent", "For Sale"])]
df = df[df["purpose"] == "For Sale"]
def convert_into_crore(text):
    if text is None or not isinstance(text, str):
        return None
    m = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)", text)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "crore":
        return num
    if unit == "lakh":
        return num / 100
    if unit == "arab":
        return num * 100
    if unit == "thousand":
        return num / 10000
    return None

df.insert(2, "price_cr", df["price"].apply(convert_into_crore))
print(df.sample(5, random_state=1)[["price", "price_cr"]])
