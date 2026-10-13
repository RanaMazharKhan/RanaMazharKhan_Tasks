# KNN exercise 4: try metric="manhattan" instead of euclidean.
# The best K changed (5 -> 16) but the best accuracy stayed the same,
# so the distance metric matters less than picking a sensible K here.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, _, y_train, _ = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
X_train_s = StandardScaler().fit_transform(X_train)

k_values = range(1, 26)
for metric in ["euclidean", "manhattan"]:
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric),
                             X_train_s, y_train, cv=5).mean() for k in k_values]
    best_k = k_values[int(np.argmax(scores))]
    print(metric, "-> best K:", best_k, "score:", round(max(scores), 4))
