# Task: plot a histogram (with KDE) of log_price.

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
rdf.loc[:, "log_price"] = np.log1p(rdf["price_cr"])
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(rdf["log_price"], kde=True)
plt.title("Distribution of log_price")
plt.xlabel("log(1 + price)")
plt.show()
