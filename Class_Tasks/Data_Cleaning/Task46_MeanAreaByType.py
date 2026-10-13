# Task: group by land_type and compute the mean area_value for each type.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
print(df.groupby("land_type")["area_value"].mean().round(2))
