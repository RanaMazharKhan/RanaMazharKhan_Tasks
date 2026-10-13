# KNN exercise 5: repeat the Iris workflow (split, scale, train, tune K)
# from scratch on the wine dataset.

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

k_values = range(1, 26)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                         X_train_s, y_train, cv=5).mean() for k in k_values]
best_k = k_values[int(np.argmax(scores))]
print("Wine: best K =", best_k, "with CV accuracy", round(max(scores), 4))

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_s, y_train)
print("Test accuracy:", round(accuracy_score(y_test, model.predict(X_test_s)), 4))
print(classification_report(y_test, model.predict(X_test_s), target_names=wine.target_names))
