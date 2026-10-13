# Task 1.3 - Split into train/test BEFORE any encoding (80/20, stratified), drop Churn columns from X.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
y = df["Churn_encoded"]
X = df.drop(columns=["Churn", "Churn_encoded"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Churn rate train:", round(y_train.mean(), 3), " test:", round(y_test.mean(), 3))

# We split first because frequency/target encoders learn from the target -
# fitting them on the full data would leak test information into training.
