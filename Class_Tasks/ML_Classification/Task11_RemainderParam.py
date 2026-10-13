# Task 2.4 - Leave one categorical column out; compare remainder='drop' vs 'passthrough'.
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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
left_out = ["PaymentMethod"]  # deliberately excluded below
used_cat = [c for c in categorical_features if c not in left_out]

def make_preprocessor(remainder):
    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), used_cat),
    ], remainder=remainder)

drop_out = make_preprocessor("drop").fit_transform(X_train)
pass_out = make_preprocessor("passthrough").fit_transform(X_train)

print("remainder='drop':", drop_out.shape)
print("remainder='passthrough':", pass_out.shape)

# 'drop' throws the left-out column away; 'passthrough' keeps it as raw text
# in the output, so the result has one extra column.
