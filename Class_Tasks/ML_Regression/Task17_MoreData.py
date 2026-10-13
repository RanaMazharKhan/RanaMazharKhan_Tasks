# Bias-variance try-it-yourself 3: change n_train from 50 to 150.
# The variance of the degree-10 model drops a lot with more data,
# because the wiggly fits average out over more points.

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def true_function(x):
    return np.sin(2 * np.pi * x)

grid = np.linspace(0, 1, 200).reshape(-1, 1)

def variance_of_degree(degree, n_train):
    preds = []
    for round_no in range(60):
        rng = np.random.RandomState(1000 + round_no)
        X = np.sort(rng.uniform(0, 1, n_train)).reshape(-1, 1)
        y = true_function(X).ravel() + rng.normal(0, 0.2, n_train)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X, y)
        preds.append(model.predict(grid))
    return float(np.mean(np.var(np.array(preds), axis=0)))

for n in [50, 150]:
    print("n_train =", n, "-> variance of degree 10:", round(variance_of_degree(10, n), 4))
print("More data lowers variance but barely changes bias.")
