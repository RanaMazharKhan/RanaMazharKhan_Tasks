# Task: plot a bar chart (figsize 5x4) of land_type value counts
# within residential_df.

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
import matplotlib.pyplot as plt

rdf["land_type"].value_counts().plot(kind="bar", figsize=(5, 4))
plt.title("Residential land types")
plt.xlabel("Land type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
