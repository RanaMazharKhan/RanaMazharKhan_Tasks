# Task - Re-run the whole classification flow end to end on the real local CSV file.
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
print("Loaded:", df.shape)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
model_df = pd.get_dummies(df.drop(columns=["customerID"]),
                          columns=df.drop(columns=["customerID"]).select_dtypes(include="object").columns,
                          drop_first=True)
X = model_df.drop(columns=["Churn"])
y = model_df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
proba = model.predict_proba(X_test_s)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))
fpr, tpr, thresholds = roc_curve(y_test, proba)
print("AUC:", round(roc_auc_score(y_test, proba), 3))

idx = np.argmax(tpr >= 0.80)
print(f"For TPR>=80%: threshold={thresholds[idx]:.3f}, FPR={fpr[idx]:.3f}")
