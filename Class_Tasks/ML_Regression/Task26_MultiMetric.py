# Grid search exercise 8: multi-metric scoring with refit="r2".
# cv_results_ now has two full sets of columns - mean_test_r2 and
# mean_test_neg_mean_squared_error (plus their ranks) - but the winner is
# refit on r2 only, because that is what refit="r2" says.

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
grid = GridSearchCV(pipe, {"knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
                           "knn__weights": ["uniform", "distance"],
                           "knn__p": [1, 2]},
                    scoring=["r2", "neg_mean_squared_error"],
                    refit="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

results = pd.DataFrame(grid.cv_results_)
print("Columns for the two metrics:")
print([c for c in results.columns if "mean_test" in c or "rank_test" in c])
print("Winner picked by r2:", grid.best_params_["knn__n_neighbors"])
