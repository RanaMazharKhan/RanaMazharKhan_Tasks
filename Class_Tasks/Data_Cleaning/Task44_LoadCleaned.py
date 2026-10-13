# Task: load dataCleansing1_complete.csv (the output of Part 1)
# with index_col=0 and preview the first rows.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
print(df.head())
print(df.shape)
