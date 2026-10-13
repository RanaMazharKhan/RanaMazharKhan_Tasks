# Task - Binary encode PaymentMethod and compare the column count with one-hot encoding.
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from category_encoders import BinaryEncoder

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
oh_cols = onehot.fit_transform(df[["PaymentMethod"]]).shape[1]

binary = BinaryEncoder(cols=["PaymentMethod"])
bin_df = binary.fit_transform(df[["PaymentMethod"]])

print("PaymentMethod values:", df["PaymentMethod"].nunique())
print("One-hot columns (drop=first):", oh_cols)
print("Binary encoded columns:", bin_df.shape[1])
print(bin_df.head())

# Binary encoding needs only 2 columns for 4 categories instead of 3, so it
# saves columns when a category has many values.
