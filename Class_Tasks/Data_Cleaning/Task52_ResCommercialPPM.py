# Task: show price-per-marla only for residential and commercial land
# types (hint: .isin) and compare them.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
residential = ["House", "Flat", "Residential Plot"]
commercial = ["Building", "Shop", "Office", "Commercial Plot"]
agricultural = ["Plot File", "Farm House", "Agricultural Land"]
industrial = ["Factory", "Industrial Land"]
tot = df.groupby("land_type")[["price_cr", "area_value"]].sum()
ppm = (tot["price_cr"] / tot["area_value"]).sort_values()
print(ppm[ppm.index.isin(residential + commercial)].round(4))
# commercial types (Building, Shop) cost more per marla than residential ones
