# Bias-variance try-it-yourself 1: change NOISE to 0.05 and 0.5.
# With tiny noise the flexible degree-15 model wins; with big noise the
# simple degree-3 model wins, because flexible models chase the noise.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def make_data(noise):
    rng = np.random.RandomState(42)
    X = np.sort(rng.uniform(0, 1, 40)).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, noise, 40)
    return X, y

for noise in [0.05, 0.2, 0.5]:
    X, y = make_data(noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    errors = {}
    for degree in [1, 3, 5, 15]:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        errors[degree] = round(mean_squared_error(y_test, model.predict(X_test)), 4)
    best = min(errors, key=errors.get)
    print("NOISE =", noise, "test MSE:", errors, "-> best degree:", best)
