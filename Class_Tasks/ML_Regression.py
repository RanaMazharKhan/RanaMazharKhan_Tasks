# ML Regression - practice tasks from the course notebooks

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, uniform, randint
from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import (train_test_split, KFold, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  SGDRegressor, Lasso)
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             mean_squared_error, r2_score, classification_report)

DATA = os.path.join(os.path.dirname(__file__), "data")
df = pd.read_csv(os.path.join(DATA, "house_prices.csv"))

# ---
# What is ML, the problem, X, y and features.
# ML means the computer learns a pattern from data instead of fixed rules.
# Our problem: predict property price (a number, so this is regression).
# X is the features table (all columns except price), y is the target (price).

small = df[["area", "price"]].dropna()
model = LinearRegression()
model.fit(small[["area"]], small["price"])  # no rule written, it learns from data
print("Learned rule: price =", round(model.coef_[0], 1), "* area +", round(model.intercept_, 1))
X_cols = df.drop(columns=["price"])
print("X = features, shape:", X_cols.shape)
print("y = target (price), average price:", round(df["price"].mean()))
print("Feature example: area, correlation with price:",
      round(df[["area", "price"]].dropna().corr().iloc[0, 1], 2))

# ---
# The ML workflow, end to end: define problem, load data, EDA, clean,
# prepare X and y, split train/test, train, evaluate.

print("1. Problem: predict property price")
print("2. Data loaded, shape:", df.shape)
model_df = df.drop(columns=["name", "agent"]).dropna()
model_df = pd.get_dummies(model_df, columns=["land_type", "location", "purpose"], drop_first=True)
print("3-4. EDA + cleaning: filled missing values, encoded categories")
X = model_df.drop(columns=["price"])
y = model_df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("5-6. X and y ready, split:", X_train.shape[0], "train rows,", X_test.shape[0], "test rows")
lin = LinearRegression()
lin.fit(X_train, y_train)
mae = mean_absolute_error(y_test, lin.predict(X_test))
print("7-8. Trained and evaluated, MAE:", round(mae))

# ---
# Why EDA comes before modeling: the data is never clean.
# We found missing values in agent, baths and area - a model cannot train
# on empty cells, so we fill or drop them before training.

missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])
print("So we fill numeric gaps with the median before training.")

# ---
# Data leakage: predicting purpose with price in the features.
# With price the model cheats (rental prices are much lower than sale
# prices) and gets 100% accuracy. Without price the score drops a lot.

df_clean = df.copy()
for col in df_clean.select_dtypes(include=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
for col in df_clean.select_dtypes(exclude=[np.number]).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

clf_df = df_clean.drop(columns=["name", "agent"])
clf_df = pd.get_dummies(clf_df, columns=["land_type", "location"], drop_first=True)
clf_df["purpose_encoded"] = LabelEncoder().fit_transform(clf_df["purpose"])

for drop_price in [False, True]:
    cols = ["purpose", "purpose_encoded"] + (["price"] if drop_price else [])
    Xl = clf_df.drop(columns=cols)
    yl = clf_df["purpose_encoded"]
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, yl, test_size=0.2, random_state=42, stratify=yl)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xl_tr, yl_tr)
    print("price dropped:", drop_price, "-> accuracy:", round(accuracy_score(yl_te, clf.predict(Xl_te)), 3))
print("Conclusion: price leaks the answer, so it must be removed.")

# ---
# KNN on Iris: try K=1, 5, 15. K=1 overfits a little and K=15 underfits a
# little, but on Iris both still score high because the classes are easy.
# Note: removing the scaler barely changes Iris accuracy because all its
# features are in cm - no feature dominates the distance.

iris = load_iris()
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)
scaler_iris = StandardScaler()
Xi_tr_s = scaler_iris.fit_transform(Xi_tr)
Xi_te_s = scaler_iris.transform(Xi_te)

for k in [1, 5, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xi_tr_s, yi_tr)
    print("K =", k, "-> test accuracy:", round(accuracy_score(yi_te, knn.predict(Xi_te_s)), 4))

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(Xi_tr, yi_tr)
print("Without scaling, accuracy:", round(accuracy_score(yi_te, knn_raw.predict(Xi_te)), 4))

# ---
# KNN tuning with cross-validation: best K with cv=5 vs cv=10.
# The best K moves (5 -> 12) because different folds give slightly
# different estimates. Trying manhattan distance instead of euclidean
# also changes the best K (5 -> 16) but the best score stays the same.

k_values = range(1, 26)
for cv in [5, 10]:
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                              Xi_tr_s, yi_tr, cv=cv).mean() for k in k_values]
    print("cv =", cv, "-> best K:", k_values[int(np.argmax(scores))],
          "score:", round(max(scores), 4))
for metric in ["euclidean", "manhattan"]:
    scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k, metric=metric),
                              Xi_tr_s, yi_tr, cv=5).mean() for k in k_values]
    print(metric, "-> best K:", k_values[int(np.argmax(scores))],
          "score:", round(max(scores), 4))

# ---
# KNN workflow from scratch on the wine dataset: split, scale, tune K.

wine = load_wine()
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42, stratify=wine.target)
Xw_tr_s = StandardScaler().fit_transform(Xw_tr)
Xw_te_s = StandardScaler().fit(Xw_tr).transform(Xw_te)

k_values = range(1, 26)
scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k),
                          Xw_tr_s, yw_tr, cv=5).mean() for k in k_values]
best_k = k_values[int(np.argmax(scores))]
print("Wine: best K =", best_k, "with CV accuracy", round(max(scores), 4))
knn_w = KNeighborsClassifier(n_neighbors=best_k)
knn_w.fit(Xw_tr_s, yw_tr)
print("Test accuracy:", round(accuracy_score(yw_te, knn_w.predict(Xw_te_s)), 4))
print(classification_report(yw_te, knn_w.predict(Xw_te_s), target_names=wine.target_names))

# ---
# Bias-variance 1: change the noise to 0.05, 0.2 and 0.5.
# With tiny noise the flexible degree-15 model wins; with big noise the
# simple degree-3 model wins, because flexible models chase the noise.

def make_sine(noise):
    rng = np.random.RandomState(42)
    X = np.sort(rng.uniform(0, 1, 40)).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, noise, 40)
    return X, y

for noise in [0.05, 0.2, 0.5]:
    Xb, yb = make_sine(noise)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=1)
    errors = {}
    for degree in [1, 3, 5, 15]:
        m = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        m.fit(Xb_tr, yb_tr)
        errors[degree] = round(mean_squared_error(yb_te, m.predict(Xb_te)), 4)
    print("NOISE =", noise, "test MSE:", errors, "-> best degree:", min(errors, key=errors.get))

# ---
# Bias-variance 2: try degrees 1, 3, 8, 20.
# Degree 3 fits the wave nicely. From degree 8 the curve starts going wild
# and test error rises again - it chases the noise.

Xb, yb = make_sine(0.2)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, random_state=1)
for degree in [1, 3, 8, 20]:
    m = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    m.fit(Xb_tr, yb_tr)
    print("Degree", degree,
          ": train MSE", round(mean_squared_error(yb_tr, m.predict(Xb_tr)), 4),
          ", test MSE", round(mean_squared_error(yb_te, m.predict(Xb_te)), 4))
print("The curve starts going wild around degree 8 (test error rises again).")

# ---
# Bias-variance 3: more data (50 vs 150 points) lowers the variance of the
# degree-10 model a lot, because the wiggly fits average out over points.

def variance_of_degree(degree, n_train):
    grid = np.linspace(0, 1, 200).reshape(-1, 1)
    preds = []
    for rnd in range(60):
        rng = np.random.RandomState(1000 + rnd)
        X = np.sort(rng.uniform(0, 1, n_train)).reshape(-1, 1)
        y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, n_train)
        m = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        m.fit(X, y)
        preds.append(m.predict(grid))
    return float(np.mean(np.var(np.array(preds), axis=0)))

for n in [50, 150]:
    print("n_train =", n, "-> variance of degree 10:", round(variance_of_degree(10, n), 4))
print("More data lowers variance but barely changes bias.")

# ---
# A decent R2 can still hide the wrong model shape. The straight line gets
# about 0.7 here, but the residuals form a wave - the line is the wrong
# shape, and judging by R2 alone would fool us.

rng = np.random.RandomState(42)
Xc = np.sort(rng.uniform(0, 3, 60)).reshape(-1, 1)
yc = 3 * Xc.ravel() + 4 * np.sin(Xc.ravel()) + rng.normal(0, 1, 60)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, random_state=1)

line = LinearRegression()
line.fit(Xc_tr, yc_tr)
print("R2 of straight line:", round(r2_score(yc_te, line.predict(Xc_te)), 3))

plt.scatter(Xc_te, yc_te - line.predict(Xc_te))
plt.axhline(0, color="red")
plt.xlabel("x")
plt.ylabel("residual (actual - predicted)")
plt.title("Residuals still curve -> straight line is the wrong shape")
plt.show()
print("The residuals form a wave, so R2 alone hid the real problem.")

# ---
# Grid search: the scoring metric picks the winner. R2 rewards fitting the
# spread of the data, MAE just averages plain errors, so switching from r2
# to neg_mean_absolute_error changes the winning k (21 -> 11).

dia = load_diabetes()
Xd_tr, _, yd_tr, _ = train_test_split(dia.data, dia.target, test_size=0.2, random_state=42)
cv5 = KFold(n_splits=5, shuffle=True, random_state=42)

pipe_knn = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsRegressor())])
grid_knn = {"knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2]}

for scoring in ["r2", "neg_mean_absolute_error"]:
    grid = GridSearchCV(pipe_knn, grid_knn, scoring=scoring, cv=cv5, n_jobs=-1)
    grid.fit(Xd_tr, yd_tr)
    print(scoring, "-> winning k:", grid.best_params_["knn__n_neighbors"],
          "score:", round(grid.best_score_, 4))

# ---
# Random search budget: re-run the SGD search with n_iter 10, 30, 60, 100, 200.
# The score jumps from 10 to 30, then stays flat - returns diminish after
# about 30 iterations because the good region is already found.

pipe_sgd = Pipeline([("sc", StandardScaler()),
                     ("sgd", SGDRegressor(random_state=42, early_stopping=False))])
dist_sgd = {
    "sgd__alpha": loguniform(1e-6, 1e1),
    "sgd__penalty": ["l1", "l2", "elasticnet"],
    "sgd__l1_ratio": uniform(0, 1),
    "sgd__learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
    "sgd__eta0": loguniform(1e-4, 1e-1),
    "sgd__max_iter": randint(1000, 8000),
}

best_scores = []
n_iters = [10, 30, 60, 100, 200]
for n in n_iters:
    search = RandomizedSearchCV(pipe_sgd, dist_sgd, n_iter=n, scoring="r2",
                                cv=cv5, n_jobs=-1, random_state=42)
    search.fit(Xd_tr, yd_tr)
    best_scores.append(search.best_score_)
    print("n_iter =", n, "-> best CV R2:", round(search.best_score_, 4))

plt.plot(n_iters, best_scores, marker="o")
plt.xlabel("n_iter (budget)")
plt.ylabel("best CV R2")
plt.title("Bigger budget helps at first, then flattens out")
plt.show()

# ---
# SGD without scaling fails. Its gradient steps are far too big for
# large-scale features and far too small for small ones - without the
# scaler the R2 collapses, so scaling is non-negotiable for SGD.

for name, pipe in [
    ("with scaler", pipe_sgd),
    ("no scaler", Pipeline([("sgd", SGDRegressor(random_state=42, early_stopping=False))])),
]:
    search = RandomizedSearchCV(pipe, dist_sgd, n_iter=20, scoring="r2",
                                cv=cv5, n_jobs=-1, random_state=42)
    search.fit(Xd_tr, yd_tr)
    print(name, "-> best CV R2:", round(search.best_score_, 4))
print("Without scaling SGD fails - scaling is non-negotiable for SGD.")

# ---
# One random search is not the truth. Running it three times with different
# random states gives the same score but different winning parameters, so
# never trust the exact params of a single search.

for rs in [0, 1, 2]:
    search = RandomizedSearchCV(pipe_sgd, dist_sgd, n_iter=60, scoring="r2",
                                cv=cv5, n_jobs=-1, random_state=rs)
    search.fit(Xd_tr, yd_tr)
    p = search.best_params_
    print("random_state =", rs,
          "| alpha:", round(p["sgd__alpha"], 4),
          "| penalty:", p["sgd__penalty"],
          "| score:", round(search.best_score_, 4))
print("Same score, different winners - one random search is not the truth.")

# ---
# Lasso with grid search over 20 alphas. The winner (alpha about 0.14)
# drove 1 of the 10 coefficients to exactly zero - that is Lasso doing
# feature selection by itself.

pipe_lasso = Pipeline([("sc", StandardScaler()),
                       ("lasso", Lasso(random_state=42, max_iter=5000))])
grid_lasso = GridSearchCV(pipe_lasso, {"lasso__alpha": np.logspace(-4, 1, 20)},
                          scoring="r2", cv=cv5, n_jobs=-1)
grid_lasso.fit(Xd_tr, yd_tr)

coef = grid_lasso.best_estimator_.named_steps["lasso"].coef_
zero = int((np.abs(coef) < 1e-10).sum())
print("best alpha:", round(grid_lasso.best_params_["lasso__alpha"], 4))
print("coefficients set to exactly zero:", zero, "out of", len(coef))
print("best CV R2:", round(grid_lasso.best_score_, 4))

# ---
# GridSearchCV by hand: nested for-loops with manual 5-fold CV give the same
# best params as GridSearchCV - it is just this loop written neatly.

best_score = -1e9
best_params = None
for k in [3, 5, 7, 9, 11, 15, 21, 31]:
    for weights in ["uniform", "distance"]:
        for p in [1, 2]:
            fold_scores = []
            for tr_idx, va_idx in cv5.split(Xd_tr):
                pipe = Pipeline([("sc", StandardScaler()),
                                 ("knn", KNeighborsRegressor(n_neighbors=k,
                                                             weights=weights, p=p))])
                pipe.fit(Xd_tr[tr_idx], yd_tr[tr_idx])
                fold_scores.append(r2_score(yd_tr[va_idx], pipe.predict(Xd_tr[va_idx])))
            mean_score = np.mean(fold_scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = (k, weights, p)
print("Manual loop best:", best_params, round(best_score, 4))

grid_check = GridSearchCV(pipe_knn, grid_knn, scoring="r2", cv=cv5, n_jobs=-1)
grid_check.fit(Xd_tr, yd_tr)
bp = grid_check.best_params_
auto_best = (bp["knn__n_neighbors"], bp["knn__weights"], bp["knn__p"])
print("GridSearchCV best:", auto_best, round(grid_check.best_score_, 4))
print("Match:", best_params == auto_best)

# ---
# Nested cross-validation: grid.best_score_ is optimistic because the same
# data picked the winner AND measured it. Nested CV re-runs the whole search
# inside fresh outer folds, so its score is lower - and more honest.

Xd_all, yd_all = dia.data, dia.target
grid_nested = GridSearchCV(pipe_knn, grid_knn, scoring="r2", cv=cv5, n_jobs=-1)

nested_scores = cross_val_score(grid_nested, Xd_all, yd_all,
                                cv=KFold(n_splits=5, shuffle=True, random_state=7),
                                scoring="r2", n_jobs=-1)
print("Nested CV R2 (honest):", round(nested_scores.mean(), 4))

grid_nested.fit(Xd_all, yd_all)
print("grid.best_score_ (optimistic):", round(grid_nested.best_score_, 4))
print("Nested is lower because the winner was chosen on unseen outer folds.")
