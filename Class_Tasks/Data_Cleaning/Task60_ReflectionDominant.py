# Task: based on the percentages, write 2-3 bullet points on which land
# types dominate the residential segment, and roughly by how much.

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
pct = rdf["land_type"].value_counts() / len(rdf) * 100
print(pct.round(1))
# - House is the biggest group with about 48% of residential listings.
# - Residential Plot is almost as big with about 47.5%.
# - Flat is small in comparison with only about 4.5%.
