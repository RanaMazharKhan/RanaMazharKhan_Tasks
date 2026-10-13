# Grid search exercise 6: grid search Lasso over 20 alphas on diabetes data.
# The winner (alpha about 0.14) drove 1 of the 10 coefficients to exactly
# zero - that is Lasso doing feature selection by itself.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

dia = load_diabetes()
X_train, _, y_train, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("sc", StandardScaler()),
                 ("lasso", Lasso(random_state=42, max_iter=5000))])
grid = GridSearchCV(pipe, {"lasso__alpha": np.logspace(-4, 1, 20)},
                    scoring="r2", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

coef = grid.best_estimator_.named_steps["lasso"].coef_
zero = int((np.abs(coef) < 1e-10).sum())
print("best alpha:", round(grid.best_params_["lasso__alpha"], 4))
print("coefficients set to exactly zero:", zero, "out of", len(coef))
print("best CV R2:", round(grid.best_score_, 4))
