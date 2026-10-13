# Grid search exercise 2: add knn__algorithm to the grid.
# The best score did not change (0.4217) because auto/ball_tree/kd_tree/brute
# all compute the exact same neighbours - only the speed differs. The grid
# got 4x bigger (128 combos), so it took longer.

import time
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

dia = load_diabetes()
X_train, _, y_train, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())])
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],
    "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
}

start = time.time()
grid = GridSearchCV(pipe, param_grid, scoring="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
print("combos:", len(grid.cv_results_["params"]), "| time:", round(time.time() - start, 1), "s")
print("best score:", round(grid.best_score_, 4))
print("best algorithm:", grid.best_params_["knn__algorithm"])
print("Score is the same as without algorithm - it only changes speed, not results.")
