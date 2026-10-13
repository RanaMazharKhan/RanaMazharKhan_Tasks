# Grid search exercise 3: re-run the SGD random search with n_iter 10 and 200,
# and plot best CV score against n_iter.
# The score jumps from 10 to 30, then stays flat - returns diminish after
# about 30 iterations because the good region is already found.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from scipy.stats import loguniform, uniform, randint

dia = load_diabetes()
X_train, _, y_train, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipe = Pipeline([("sc", StandardScaler()),
                 ("sgd", SGDRegressor(random_state=42, early_stopping=False))])
param_dist = {
    "sgd__alpha": loguniform(1e-6, 1e1),
    "sgd__penalty": ["l1", "l2", "elasticnet"],
    "sgd__l1_ratio": uniform(0, 1),
    "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "sgd__eta0": loguniform(1e-4, 1e-1),
    "sgd__max_iter": randint(1000, 8000),
}

n_iters = [10, 30, 60, 100, 200]
best_scores = []
for n in n_iters:
    search = RandomizedSearchCV(pipe, param_dist, n_iter=n, scoring="r2",
                                cv=cv, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    best_scores.append(search.best_score_)
    print("n_iter =", n, "-> best CV R2:", round(search.best_score_, 4))

plt.plot(n_iters, best_scores, marker="o")
plt.xlabel("n_iter (budget)")
plt.ylabel("best CV R2")
plt.title("Bigger budget helps at first, then flattens out")
plt.show()
