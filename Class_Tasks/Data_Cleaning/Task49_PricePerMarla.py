# Task: for each land_type compute total price_cr and total area_value,
# then price-per-marla = total price / total area. Sort and plot as bar chart.

import os
import pandas as pd
import numpy as np

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "dataCleansing1_complete.csv"), index_col=0)
tot = df.groupby("land_type")[["price_cr", "area_value"]].sum()
ppm = (tot["price_cr"] / tot["area_value"]).sort_values()
print(ppm.round(4))
import matplotlib.pyplot as plt

ppm.plot(kind="bar")
plt.title("Price per marla by land type")
plt.ylabel("Price per marla (Crore)")
plt.tight_layout()
plt.show()
