# Task: plot a boxplot of log_price. Did the log transform help reduce
# the spread of outliers compared to raw price_cr?

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

sns.boxplot(x=rdf["log_price"])
plt.title("Boxplot of log_price")
plt.show()
# yes - the log transform pulls the extreme values in, the box looks
# much more compact than the raw price_cr boxplot
