# Task: plot a boxplot of price_cr grouped by land_type (x=land_type,
# y=price_cr), x labels rotated 90, log-scaled y-axis, with title and labels.

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
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x="land_type", y="price_cr", data=rdf)
plt.xticks(rotation=90)
plt.yscale("log")
plt.title("Price by land type")
plt.xlabel("Land type")
plt.ylabel("Price (Crore, log scale)")
plt.tight_layout()
plt.show()
