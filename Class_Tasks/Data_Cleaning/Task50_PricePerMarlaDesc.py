# Task: display the price-per-marla series sorted in descending order.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
tot = df.groupby("land_type")[["price_cr", "area_value"]].sum()
ppm = (tot["price_cr"] / tot["area_value"]).sort_values()
print(ppm.sort_values(ascending=False).round(4))
