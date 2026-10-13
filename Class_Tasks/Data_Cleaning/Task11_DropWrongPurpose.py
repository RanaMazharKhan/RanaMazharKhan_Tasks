# Task: drop the rows with invalid purpose values using their index.

import os
import re
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})
df = df.drop_duplicates()
df = df.drop(columns=["agent", "name"])
wrong_scrap = df[~df["purpose"].isin(["For Rent", "For Sale"])]
df = df.drop(index=wrong_scrap.index)
print("Shape after dropping:", df.shape)
