# Homework 3 - RandomizedSearchCV (50 tries) then a fine GridSearchCV; report the improvement.
from scipy.stats import randint
from sklearn.datasets import load_wine
from sklearn.model_selection import (train_test_split, RandomizedSearchCV,
                                     GridSearchCV)
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
baseline.fit(X_train, y_train)
base_acc = baseline.score(X_test, y_test)
print("Baseline test accuracy:", round(base_acc, 4))

rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": randint(100, 400),
     "max_depth": [3, 5, 8, 12, None],
     "min_samples_leaf": randint(1, 10),
     "max_features": ["sqrt", "log2", 0.5, 1.0]},
    n_iter=50, cv=3, random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
print("Random search best:", rand_search.best_params_)

best = rand_search.best_params_
grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [best["n_estimators"] - 50, best["n_estimators"], best["n_estimators"] + 50],
     "max_depth": [d for d in [best["max_depth"] - 2, best["max_depth"], best["max_depth"] + 2]
                   if d is None or d > 0],
     "min_samples_leaf": [max(1, best["min_samples_leaf"] - 1), best["min_samples_leaf"],
                          best["min_samples_leaf"] + 1]},
    cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

final_acc = grid.best_estimator_.score(X_test, y_test)
print("Fine grid best:", grid.best_params_)
print("Final test accuracy:", round(final_acc, 4))
print("Improvement over baseline:", round(final_acc - base_acc, 4))
