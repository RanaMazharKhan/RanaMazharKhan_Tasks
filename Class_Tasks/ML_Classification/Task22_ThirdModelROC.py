# Task - Train a third model (GradientBoostingClassifier) and add its ROC curve to the comparison.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
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

lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_s, y_train)
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(X_train, y_train)
gb = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

plt.figure(figsize=(6.5, 6.5))
for name, proba, color in [
    ("Logistic Regression", lr.predict_proba(X_test_s)[:, 1], "#1565c0"),
    ("Random Forest", rf.predict_proba(X_test)[:, 1], "#2e7d32"),
    ("Gradient Boosting", gb.predict_proba(X_test)[:, 1], "#ef6c00"),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    print(f"{name}: AUC = {roc_auc_score(y_test, proba):.3f}")
    plt.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC = {roc_auc_score(y_test, proba):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Model Comparison via ROC Curve")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
