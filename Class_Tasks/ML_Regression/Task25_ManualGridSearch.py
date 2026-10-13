# Grid search exercise 7: write GridSearchCV by hand with nested for-loops
# and manual 5-fold CV, and confirm we get the same best params.
# Same answer - GridSearchCV is just this loop written neatly.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

dia = load_diabetes()
X_train, _, y_train, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = -1e9
best_params = None
for k in [3, 5, 7, 9, 11, 15, 21, 31]:
    for weights in ["uniform", "distance"]:
        for p in [1, 2]:
            fold_scores = []
            for train_idx, val_idx in cv.split(X_train):
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("knn", KNeighborsRegressor(n_neighbors=k,
                                                             weights=weights, p=p))])
                pipe.fit(X_train[train_idx], y_train[train_idx])
                fold_scores.append(r2_score(y_train[val_idx], pipe.predict(X_train[val_idx])))
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = (k, weights, p)
print("Manual loop best:", best_params, round(best_score, 4))

pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())])
grid = GridSearchCV(pipe, {"knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                           "knn__weights": ["uniform", "distance"],
                           "knn__p": [1, 2]},
                    scoring="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
bp = grid.best_params_
print("GridSearchCV best:", (bp["knn__n_neighbors"], bp["knn__weights"], bp["knn__p"]),
      round(grid.best_score_, 4))
print("Match:", best_params == (bp["knn__n_neighbors"], bp["knn__weights"], bp["knn__p"]))
