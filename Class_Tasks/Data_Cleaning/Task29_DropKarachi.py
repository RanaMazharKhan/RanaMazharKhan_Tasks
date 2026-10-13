# Task: this dataset should only have Lahore properties - find rows where
# location contains 'Karachi' and drop them.

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
group = df[(df["location"] == "LDA Road, Lahore, Punjab") & (df["land_type"] == "Residential Plot") & (df["area"] == "5 Marla")]
mean_price = round(group["price_cr"].mean(), 2)
df.at[18903, "price_cr"] = mean_price
def get_loc(loc):
    return loc.replace(", Lahore, Punjab", "")

df.insert(4, "locat", df["location"].apply(get_loc))
def convert_into_marla(text):
    m = re.search(r"(\d+\.?\d*)", text)
    if not m:
        return None
    val = float(m.group(1))
    t = text.lower()
    if "kanal" in t:
        return val * 20
    if "sq" in t or "yd" in t:
        return val * 9 / 225
    return val

df.insert(7, "area_value", df["area"].apply(convert_into_marla))
print("Karachi rows:", df["location"].str.contains("Karachi", na=False).sum())
df = df[~df["location"].str.contains("Karachi", na=False)]
print("Shape after dropping:", df.shape)
