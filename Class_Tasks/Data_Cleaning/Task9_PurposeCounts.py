# Task: check the value counts of 'purpose'. Are there values other
# than 'For Rent' and 'For Sale'?

import os
import re
import pandas as pd
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})
df = df.drop_duplicates()
df = df.drop(columns=["agent", "name"])
print(df["purpose"].value_counts())
# yes - some rows have addresses in the purpose column (scraping error)
