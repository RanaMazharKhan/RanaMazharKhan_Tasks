# Task 1.2 - Label encode the target column Churn (Yes/No) into Churn_encoded and print the mapping.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))

le = LabelEncoder()
df["Churn_encoded"] = le.fit_transform(df["Churn"])

print("0 =", le.classes_[0])
print("1 =", le.classes_[1])
print(df[["Churn", "Churn_encoded"]].head())
