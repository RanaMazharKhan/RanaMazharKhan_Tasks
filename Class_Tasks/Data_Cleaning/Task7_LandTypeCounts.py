# Task: look at the value counts of land_type to see the categories.

import os
import re
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})
df = df.drop_duplicates()
print(df["land_type"].value_counts())
