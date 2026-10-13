# Grid search exercise 1: change scoring from r2 to neg_mean_absolute_error.
# Yes, the winning k changed (21 -> 11). R2 rewards fitting the spread of
# the data, MAE just averages plain errors, so they pick different winners.

import numpy as np
import pandas as pd
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
}

for scoring in ["r2", "neg_mean_absolute_error"]:
    grid = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(scoring, "-> winning k:", grid.best_params_["knn__n_neighbors"],
          "score:", round(grid.best_score_, 4))
