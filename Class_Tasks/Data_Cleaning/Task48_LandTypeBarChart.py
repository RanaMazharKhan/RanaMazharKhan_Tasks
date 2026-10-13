# Task: plot a bar chart of the land_type value counts.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
import matplotlib.pyplot as plt

df["land_type"].value_counts().plot(kind="bar")
plt.title("Listings by land type")
plt.xlabel("Land type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
