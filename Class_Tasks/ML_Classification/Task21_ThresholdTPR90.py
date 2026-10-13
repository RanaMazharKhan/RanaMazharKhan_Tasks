# Task - Set target_tpr to 0.90: how much does FPR rise vs 0.80, and what happens to precision?
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_score

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
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
y_proba = model.predict_proba(X_test_s)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

for target in [0.80, 0.90]:
    idx = np.argmax(tpr >= target)
    thr, at_tpr, at_fpr = thresholds[idx], tpr[idx], fpr[idx]
    prec = precision_score(y_test, (y_proba >= thr).astype(int))
    print(f"TPR>={target}: threshold={thr:.3f}, FPR={at_fpr:.3f}, precision={prec:.3f}")

# Catching 90% of churners instead of 80% forces a lower threshold, so FPR
# rises (more loyal customers wrongly flagged) and precision drops because
# more of our "will churn" predictions are wrong.
