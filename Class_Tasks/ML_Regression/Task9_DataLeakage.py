# Day 9 try-it-yourself: drop price when predicting purpose.
# With price in the features the model cheats (rental prices are much lower
# than sale prices) and gets 100% accuracy. Without price it drops a lot -
# that is data leakage.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

df_clean = df.copy()
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
for col in df_clean.select_dtypes(exclude=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

clf_df = df_clean.drop(columns=["name", "agent"])
clf_df = pd.get_dummies(clf_df, columns=["land_type", "location"], drop_first=True)
clf_df["purpose_encoded"] = LabelEncoder().fit_transform(clf_df["purpose"])

for drop_price in [False, True]:
    cols = ["purpose", "purpose_encoded"] + (["price"] if drop_price else [])
    X = clf_df.drop(columns=cols)
    y = clf_df["purpose_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print("price dropped:" , drop_price, "-> accuracy:", round(acc, 3))

print("Conclusion: price leaks the answer, so it must be removed.")
