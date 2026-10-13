# KNN exercise 3: change cv=5 to cv=10 in the elbow search. Does best K change?
# Yes - with cv=10 the best K moved to 12 (it was 5 with cv=5), because
# different folds give slightly different estimates of the best K.

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
for cv in [5, 10]:
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             X_train_s, y_train, cv=cv).mean() for k in k_values]
    best_k = k_values[int(np.argmax(scores))]
    print("cv =", cv, "-> best K:", best_k, "score:", round(max(scores), 4))
