# Data Cleaning and EDA of the Zameen.com property listings dataset.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---
# Load the CSV and fix the misspelled 'pupose' column.
DATA = os.path.join(os.path.dirname(__file__), "data")
df = pd.read_csv(os.path.join(DATA, "zameenData.csv"), index_col=0)
df = df.rename(columns={"pupose": "purpose"})

# ---
# Drop exact duplicate rows.
df = df.drop_duplicates()

# ---
# Drop 'agent' and 'name' - they don't help predict prices.
df = df.drop(columns=["agent", "name"])

# ---
# Some rows have addresses scraped into 'purpose' - keep only valid rows.
df = df[df["purpose"].isin(["For Rent", "For Sale"])]

# ---
# Keep only 'For Sale' listings.
df = df[df["purpose"] == "For Sale"]

# ---
# Parse prices like 'PKR1.2 Crore' into numbers (unit: Crore).
def convert_into_crore(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"(\d+\.?\d*)\s*([a-zA-Z]+)", text)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "crore":
        return num
    if unit == "lakh":
        return num / 100
    if unit == "arab":
        return num * 100
    if unit == "thousand":
        return num / 10000
    return None

df.insert(2, "price_cr", df["price"].apply(convert_into_crore))

# ---
# Fix the wrong price at index 18903 with the mean of similar listings
# (LDA Road, Residential Plot, 5 Marla).
group = df[(df["location"] == "LDA Road, Lahore, Punjab") &
           (df["land_type"] == "Residential Plot") &
           (df["area"] == "5 Marla")]
df.at[18903, "price_cr"] = round(group["price_cr"].mean(), 2)

# ---
# Extract the short area name from location (drop ', Lahore, Punjab').
def get_loc(loc):
    return loc.replace(", Lahore, Punjab", "")

df.insert(4, "locat", df["location"].apply(get_loc))

# ---
# Parse areas like '2 Kanal' into numbers (unit: Marla).
# 1 Kanal = 20 Marla, 1 Sq. Yd. = 9/225 Marla.
def convert_into_marla(text):
    m = re.search(r"(\d+\.?\d*)", text)
    if not m:
        return None
    val = float(m.group(1))
    t = text.lower()
    if "kanal" in t:
        return val * 20
    if "sq" in t or "yd" in t:
        return val * 9 / 225
    return val

df.insert(7, "area_value", df["area"].apply(convert_into_marla))

# ---
# This dataset is about Lahore - drop rows from other cities.
df = df[~df["location"].str.contains("Karachi", na=False)]

# ---
# Replace "-" with "0" in baths and bed, and make area_value numeric.
df["baths"] = df["baths"].replace("-", "0")
df["bed"] = df["bed"].replace("-", "0")
df["area_value"] = df["area_value"].astype(float)

# ---
# Keep only the land types we will actually model.
useful = ["House", "Flat", "Building", "Residential Plot", "Commercial Plot",
          "Plot File", "Agricultural Land", "Industrial Land", "Factory",
          "Farm House", "Shop", "Office"]
df = df[df["land_type"].isin(useful)]
print(df.shape)

# ---
# Check for missing values in every column.
print(df.isnull().sum())

# ---
# Price per marla for each land type, sorted, with a bar chart.
tot = df.groupby("land_type")[["price_cr", "area_value"]].sum()
ppm = (tot["price_cr"] / tot["area_value"]).sort_values()
print(ppm.round(4))
ppm.plot(kind="bar")
plt.title("Price per marla by land type")
plt.ylabel("Price per marla (Crore)")
plt.tight_layout()
plt.show()

# ---
# Group land types into 4 categories and compare price per marla.
residential = ["House", "Flat", "Residential Plot"]
commercial = ["Building", "Shop", "Office", "Commercial Plot"]
agricultural = ["Plot File", "Farm House", "Agricultural Land"]
industrial = ["Factory", "Industrial Land"]
print(ppm[ppm.index.isin(residential)].round(4))
print(ppm[ppm.index.isin(commercial)].round(4))
print(ppm[ppm.index.isin(agricultural)].round(4))
print(ppm[ppm.index.isin(industrial)].round(4))
# commercial types (Building, Shop) cost more per marla than residential ones

# ---
# Focus on residential listings; drop raw columns and rename 'locat'.
rdf = df[df["land_type"].isin(residential)].copy()
rdf = rdf.drop(columns=["location", "price", "area", "added_days", "purpose"])
rdf = rdf.rename(columns={"locat": "location"})

# ---
# Group rare locations (fewer than 40 listings) into "Others".
rare = rdf["location"].value_counts()
rare = rare[rare < 40].index
rdf.loc[:, "location"] = rdf["location"].replace(rare, "Others")
print("Distinct locations now:", rdf["location"].nunique())

# ---
# Encode location by its mean price (target encoding).
rdf.loc[:, "location_encoded"] = rdf["location"].map(
    rdf.groupby("location")["price_cr"].mean())

# ---
# Summary stats of price. Mean is much bigger than median - a few very
# expensive listings pull the average up, so prices are right-skewed.
print(rdf["price_cr"].describe().round(2))

# ---
# Histogram of prices.
sns.histplot(rdf["price_cr"], bins=100, kde=True)
plt.title("Distribution of price_cr")
plt.xlabel("Price (Crore)")
plt.show()

# ---
# Same histogram with a log-scaled x-axis.
sns.histplot(rdf["price_cr"], bins=100, kde=True, log_scale=True)
plt.title("Distribution of price_cr (log x-axis)")
plt.xlabel("Price (Crore)")
plt.show()

# ---
# Boxplot of prices to see the outliers.
sns.boxplot(x=rdf["price_cr"])
plt.title("Boxplot of price_cr")
plt.show()

# ---
# Measure the skew and kurtosis of prices.
print("Skewness:", round(rdf["price_cr"].skew(), 2))
print("Kurtosis:", round(rdf["price_cr"].kurt(), 2))

# ---
# Log transform the prices to shrink the extreme values.
rdf.loc[:, "log_price"] = np.log1p(rdf["price_cr"])

# ---
# Boxplot and histogram of log prices. The log transform pulls the
# extreme values in - the box looks much more compact now.
sns.boxplot(x=rdf["log_price"])
plt.title("Boxplot of log_price")
plt.show()
sns.histplot(rdf["log_price"], kde=True)
plt.title("Distribution of log_price")
plt.xlabel("log(1 + price)")
plt.show()

# ---
# Zoom into Residential Plots and compute price per marla.
rplot = rdf[rdf["land_type"] == "Residential Plot"].copy()
rplot.loc[:, "price_per_marla"] = (
    rplot["price_cr"] / rplot["area_value"]).round(2)

# ---
# Histogram and boxplot of price per marla.
sns.histplot(rplot["price_per_marla"], bins=50, kde=True)
plt.title("Distribution of price per marla (Residential Plot)")
plt.xlabel("Price per marla (Crore)")
plt.show()
sns.boxplot(x=rplot["price_per_marla"])
plt.title("Boxplot of price per marla")
plt.show()

# ---
# Check rows with suspicious price_per_marla >= 1.
out = rplot[rplot["price_per_marla"] >= 1]
print(out[["price_cr", "area_value", "price_per_marla", "location"]].to_string())
# one row has area_value 0 which makes price_per_marla infinite - a data error

# ---
# Drop nulls (there are none) and check the spread of price per marla.
rplot = rplot.dropna()
print("Std of price_per_marla:",
      rplot["price_per_marla"].replace([float("inf")], float("nan")).std().round(4))

# ---
# Keep only the columns needed for modeling.
rdf = rdf.drop(columns=["land_type", "location_encoded", "log_price", "baths", "bed"])
print(rdf.columns.tolist())
print(rdf.shape)
