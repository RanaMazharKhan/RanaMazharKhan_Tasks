# Task - Remove SimpleImputer from the numeric branch and see what error .fit() gives.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "Telco-Customer-Churn.csv"))
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # still has NaNs
df["Churn_encoded"] = LabelEncoder().fit_transform(df["Churn"])
X = df.drop(columns=["Churn", "Churn_encoded", "customerID"])
y = df["Churn_encoded"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X_train.columns if c not in numeric_features]

broken = ColumnTransformer(transformers=[
    ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),  # no imputer!
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
], remainder="drop")

pipe = Pipeline(steps=[
    ("preprocessor", broken),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
])

try:
    pipe.fit(X_train, y_train)
except ValueError as e:
    print("Error:", e)

# The scaler chokes on the NaN values in TotalCharges - this is exactly the
# bug the SimpleImputer step in a Pipeline is designed to prevent.
