# Task: load zameenData.csv with index_col=0, rename the typo column
# 'pupose' to 'purpose', and show the first 5 rows.

import os
import re
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})
print(df.head())
