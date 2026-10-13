# Task - Final mini-project: best churn pipeline, tune it, report results, save with joblib.
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
], remainder="drop")

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42)),
])

grid = GridSearchCV(pipe,
                    {"classifier__n_estimators": [200, 400],
                     "classifier__max_depth": [8, 12, None]},
                    cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

best = grid.best_estimator_
pred = best.predict(X_test)
print("Best params:", grid.best_params_)
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))

joblib.dump(best, os.path.join(os.path.dirname(__file__), "my_churn_pipeline.pkl"))
print("Saved my_churn_pipeline.pkl")

# Write-up:
# I one-hot encoded the nominal columns (gender, InternetService, etc.) because
# trees and linear models need numbers and one-hot avoids fake ordering, while
# the numeric columns only needed median imputation plus scaling.
# ColumnTransformer + Pipeline made this much easier than doing it by hand:
# one object remembers every step, the same preprocessing is applied to the
# test set automatically, and GridSearchCV can tune the model without any
# data leaking between folds.
