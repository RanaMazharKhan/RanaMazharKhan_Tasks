# Task 1.7 - Target encode PaymentMethod (mean of y_train per group) using train data only.
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

train_with_y = X_train.copy()
train_with_y["Churn"] = y_train.values
target_means = train_with_y.groupby("PaymentMethod")["Churn"].mean()
print(target_means.sort_values(ascending=False).round(3))

X_copy = X_train.copy()
X_copy["PaymentMethod_target"] = X_copy["PaymentMethod"].map(target_means)
print(X_copy[["PaymentMethod", "PaymentMethod_target"]].head())

# If we used the FULL dataset, the means would be computed with the test
# labels too - the model would peek at answers it should never see (leakage),
# so test scores would look better than they really are.
