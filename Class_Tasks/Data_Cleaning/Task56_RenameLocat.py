# Task: rename the 'locat' column in df1 to 'location'.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
df1 = df.drop(columns=["location", "price", "area", "added_days", "purpose"])
df1 = df1.rename(columns={"locat": "location"})
print(df1.columns.tolist())
