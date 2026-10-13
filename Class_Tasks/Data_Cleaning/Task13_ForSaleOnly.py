# Task: keep only rows where purpose is 'For Sale' and check the shape.

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
print(df.shape)
