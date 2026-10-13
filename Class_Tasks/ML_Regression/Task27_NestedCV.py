# Grid search exercise 9: nested cross-validation.
# grid.best_score_ is optimistic because the same data picked the winner AND
# measured it. Nested CV re-runs the whole search inside fresh outer folds,
# so its score is lower - and more honest about new data.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

dia = load_diabetes()
X, y = dia.data, dia.target
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())])
param_grid = {"knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
              "knn__weights": ["uniform", "distance"],
              "knn__p": [1, 2]}
grid = GridSearchCV(pipe, param_grid, scoring="r2", cv=cv, n_jobs=-1)

nested_scores = cross_val_score(grid, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=7),
                                scoring="r2", n_jobs=-1)
print("Nested CV R2 (honest):", round(nested_scores.mean(), 4))

grid.fit(X, y)
print("grid.best_score_ (optimistic):", round(grid.best_score_, 4))
print("Nested is lower because the winner was chosen on unseen outer folds.")
