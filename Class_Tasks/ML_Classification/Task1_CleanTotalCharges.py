# Task 1.1 - Clean TotalCharges: convert it from text to numbers and count missing values.
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))

print("dtype before:", df["TotalCharges"].dtype)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("dtype after:", df["TotalCharges"].dtype)
print("Missing values after conversion:", df["TotalCharges"].isna().sum())
