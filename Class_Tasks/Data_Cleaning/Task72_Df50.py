# Task: filter rows where price_cr > 50 into df_50, then compute
# price_per_marla = price_cr / area_value for that subset.

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
rdf = df1[df1["land_type"].isin(residential)].copy()
rare = rdf["location"].value_counts()
rare = rare[rare < 40].index
rdf.loc[:, "location"] = rdf["location"].replace(rare, "Others")
rdf.loc[:, "location_encoded"] = rdf["location"].map(rdf.groupby("location")["price_cr"].mean())
df_50 = rdf[rdf["price_cr"] > 50].copy()
df_50["price_per_marla"] = df_50["price_cr"] / df_50["area_value"]
print(len(df_50))
print(df_50[["price_cr", "area_value", "price_per_marla"]].head())
