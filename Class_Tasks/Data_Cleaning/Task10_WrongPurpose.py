# Task: select rows where purpose is neither 'For Rent' nor 'For Sale'
# and store them in wrong_scrap.

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
print("Bad rows:", len(wrong_scrap))
print(wrong_scrap["purpose"].value_counts().head())
