# Task: check the value counts of land_type.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
print(df["land_type"].value_counts())
