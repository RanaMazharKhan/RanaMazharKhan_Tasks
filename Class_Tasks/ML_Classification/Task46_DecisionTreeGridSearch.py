# Homework 3 - Tune a decision tree on wine with GridSearchCV and report the best parameters.
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {"criterion": ["gini", "entropy"],
     "max_depth": [2, 3, 4, 5, 7, 10, None],
     "min_samples_split": [2, 5, 10],
     "min_samples_leaf": [1, 2, 5]},
    cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV score:", round(grid.best_score_, 4))
print("Test score:", round(grid.best_estimator_.score(X_test, y_test), 4))
