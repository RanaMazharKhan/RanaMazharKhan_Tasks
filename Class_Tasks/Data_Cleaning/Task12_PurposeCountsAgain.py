# Task: check purpose value counts again - only For Rent/For Sale should remain.

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
print(df["purpose"].value_counts())
