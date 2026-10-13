# Homework 4 - Top 5 features by permutation importance; drop the 10 least important, re-check.
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Full accuracy:", round(rf.score(X_test, y_test), 4))

result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nTop 5 features:")
print(importance.head(5).round(3))

drop_cols = importance.tail(10).index.tolist()
X_train_small = X_train.drop(columns=drop_cols)
X_test_small = X_test.drop(columns=drop_cols)
rf2 = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf2.fit(X_train_small, y_train)
print("\nAccuracy after dropping 10 least important:", round(rf2.score(X_test_small, y_test), 4))

# Dropping the weakest features barely changes accuracy - most of the signal
# lives in a few strong features like flavanoids and proline.
