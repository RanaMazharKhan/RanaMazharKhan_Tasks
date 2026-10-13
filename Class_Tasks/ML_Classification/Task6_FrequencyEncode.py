# Task 1.6 - Frequency encode PaymentMethod manually with pandas, using X_train only.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
X = df.drop(columns=["Churn", "Churn_encoded"])
y = df["Churn_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# frequency map learned from the training data only
freq_map = X_train["PaymentMethod"].value_counts(normalize=True)
print(freq_map.round(3))

X_copy = X_train.copy()
X_copy["PaymentMethod_freq"] = X_copy["PaymentMethod"].map(freq_map)
print(X_copy[["PaymentMethod", "PaymentMethod_freq"]].head())
