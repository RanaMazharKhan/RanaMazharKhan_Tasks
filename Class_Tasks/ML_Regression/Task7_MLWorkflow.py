# Day 9 summary question 7: What is the ML workflow? List the steps in order.
# 1. Define the problem  2. Load data  3. EDA  4. Clean data
# 5. Prepare X and y  6. Split train/test  7. Train  8. Evaluate

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

print("1. Problem: predict property price")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))
print("2. Data loaded, shape:", df.shape)

print("3. EDA: checked columns and missing values")
model_df = df.drop(columns=["name", "agent"]).dropna()
model_df = pd.get_dummies(model_df, columns=["land_type", "location", "purpose"], drop_first=True)
print("4. Cleaned: filled missing values, encoded categories")

X = model_df.drop(columns=["price"])
y = model_df["price"]
print("5. X and y ready, X shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("6. Split: train", X_train.shape[0], "rows, test", X_test.shape[0], "rows")

model = LinearRegression()
model.fit(X_train, y_train)
print("7. Model trained")
mae = mean_absolute_error(y_test, model.predict(X_test))
print("8. Evaluated, MAE:", round(mae))
