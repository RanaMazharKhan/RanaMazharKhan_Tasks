# Task: compute the standard deviation of price_per_marla after cleanup.

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
residential_plot_df = rdf[rdf["land_type"] == "Residential Plot"].copy()
residential_plot_df.loc[:, "price_per_marla"] = (residential_plot_df["price_cr"] / residential_plot_df["area_value"]).round(2)
residential_plot_df = residential_plot_df.dropna()
print("Std of price_per_marla:", residential_plot_df["price_per_marla"].replace([float("inf")], float("nan")).std().round(4))
# there are no nulls, so nothing was dropped; one row has area 0 which
# makes its price_per_marla infinite (a data error)
