# Task 1.4 - One-hot encode gender and InternetService of X_train, print shape and column names.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
X = df.drop(columns=["Churn", "Churn_encoded"])
y = df["Churn_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
encoded = enc.fit_transform(X_train[["gender", "InternetService"]])

print("Shape:", encoded.shape)
print("Column names:", list(enc.get_feature_names_out(["gender", "InternetService"])))
