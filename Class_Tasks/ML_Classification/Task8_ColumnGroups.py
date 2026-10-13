# Task 2.1 - Define numeric_features and derive categorical_features from X_train columns.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
X = df.drop(columns=["Churn", "Churn_encoded", "customerID"])
y = df["Churn_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

print("Numeric:", numeric_features)
print("Categorical:", categorical_features)
