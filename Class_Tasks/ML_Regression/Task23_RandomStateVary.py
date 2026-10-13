# Grid search exercise 5: run the random search three times with
# random_state 0, 1 and 2.
# The winning hyper-parameters jump around (alpha 0.0 vs 0.25 vs 0.61,
# elasticnet vs l1) while the score barely moves. So never trust the exact
# params of one random search - the score is stable, the params are not.

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

for rs in [0, 1, 2]:
    search = RandomizedSearchCV(pipe, param_dist, n_iter=60, scoring="r2",
                                cv=cv, n_jobs=-1, random_state=rs)
    search.fit(X_train, y_train)
    p = search.best_params_
    print("random_state =", rs,
          "| alpha:", round(p["sgd__alpha"], 4),
          "| penalty:", p["sgd__penalty"],
          "| score:", round(search.best_score_, 4))
print("Same score, different winners - one random search is not the truth.")
