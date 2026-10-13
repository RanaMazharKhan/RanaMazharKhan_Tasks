# Task: split df1 into residential_df, commercial_df, agricultural_df
# and industrial_df using the category lists.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
residential = ["House", "Flat", "Residential Plot"]
commercial = ["Building", "Shop", "Office", "Commercial Plot"]
agricultural = ["Plot File", "Farm House", "Agricultural Land"]
industrial = ["Factory", "Industrial Land"]
df1 = df.drop(columns=["location", "price", "area", "added_days", "purpose"])
df1 = df1.rename(columns={"locat": "location"})
residential_df = df1[df1["land_type"].isin(residential)]
commercial_df = df1[df1["land_type"].isin(commercial)]
agricultural_df = df1[df1["land_type"].isin(agricultural)]
industrial_df = df1[df1["land_type"].isin(industrial)]
print(len(residential_df), len(commercial_df), len(agricultural_df), len(industrial_df))
