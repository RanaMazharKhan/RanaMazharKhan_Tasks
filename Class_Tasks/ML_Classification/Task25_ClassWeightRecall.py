# Task - Add class_weight='balanced' to LogisticRegression and check if churn recall improves.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

for weight in [None, "balanced"]:
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight=weight)),
    ])
    pipe.fit(X_train, y_train)
    report = classification_report(y_test, pipe.predict(X_test), output_dict=True)
    print(f"class_weight={weight}: churn recall = {report['1']['recall']:.3f}, "
          f"churn precision = {report['1']['precision']:.3f}")

# Balanced weights push the model to catch more churners (higher recall),
# usually at the cost of a little precision.
