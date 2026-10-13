# KNN exercise 1: try K=1 and K=15 on Iris. What happens to test accuracy?
# K=1 overfits a little and K=15 underfits a little, but on Iris both still
# score high because the classes are easy to separate.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

for k in [1, 5, 15]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s))
    print("K =", k, "-> test accuracy:", round(acc, 4))
