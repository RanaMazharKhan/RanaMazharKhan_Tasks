# KNN exercise 2: remove StandardScaler and re-run. Does accuracy change?
# On Iris barely - all features are in cm so no feature dominates. On the
# fruit example (grams vs a 1-10 score) weight would dominate the distance
# and scaling would matter a lot more.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

model_raw = KNeighborsClassifier(n_neighbors=5)
model_raw.fit(X_train, y_train)
print("Without scaling, accuracy:", round(accuracy_score(y_test, model_raw.predict(X_test)), 4))

scaler = StandardScaler()
model_scaled = KNeighborsClassifier(n_neighbors=5)
model_scaled.fit(scaler.fit_transform(X_train), y_train)
print("With scaling,    accuracy:", round(accuracy_score(y_test, model_scaled.predict(scaler.transform(X_test))), 4))
print("Iris features are all in cm, so scaling changes almost nothing here.")
