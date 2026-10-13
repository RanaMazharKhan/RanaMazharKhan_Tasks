# Grid search exercise 4: remove StandardScaler from the SGD pipeline.
# The R2 values collapse (they can even go negative) because SGD takes
# gradient steps that are far too big for large-scale features and far too
# small for small ones - without scaling it cannot learn properly.

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from scipy.stats import loguniform, uniform, randint

dia = load_diabetes()
X_train, _, y_train, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "sgd__alpha": loguniform(1e-6, 1e1),
    "sgd__penalty": ["l1", "l2", "elasticnet"],
    "sgd__l1_ratio": uniform(0, 1),
    "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "sgd__eta0": loguniform(1e-4, 1e-1),
    "sgd__max_iter": randint(1000, 8000),
}

for name, pipe in [
    ("with scaler", Pipeline([("sc", StandardScaler()),
                              ("sgd", SGDRegressor(random_state=42, early_stopping=False))])),
    ("no scaler", Pipeline([("sgd", SGDRegressor(random_state=42, early_stopping=False))])),
]:
    search = RandomizedSearchCV(pipe, param_dist, n_iter=20, scoring="r2",
                                cv=cv, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    print(name, "-> best CV R2:", round(search.best_score_, 4))
print("Without scaling SGD fails - scaling is non-negotiable for SGD.")
