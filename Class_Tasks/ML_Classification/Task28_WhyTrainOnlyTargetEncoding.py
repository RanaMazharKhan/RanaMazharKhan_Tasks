# Task - Show why target encoding must use X_train/y_train only, never the full dataset.
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

train_only = X_train.copy()
train_only["Churn"] = y_train.values
train_means = train_only.groupby("PaymentMethod")["Churn"].mean()

full = pd.concat([X_train, X_test])
full_y = pd.concat([y_train, y_test])
full_with_y = full.copy()
full_with_y["Churn"] = full_y.values
full_means = full_with_y.groupby("PaymentMethod")["Churn"].mean()

print("Train-only means:\n", train_means.round(4))
print("\nFull-data means:\n", full_means.round(4))
print("\nDifference:\n", (full_means - train_means).round(4))

# PaymentMethod_target must be built from X_train/y_train only, because the
# means are computed from the target itself. Using the full dataset lets the
# test labels sneak into the training features (data leakage): the model gets
# hints about the answers, so test scores look better than they really are.
