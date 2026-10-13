# Bias-variance try-it-yourself 2: try degrees [1, 3, 8, 20].
# Degree 3 fits the wave nicely. From degree 8 onwards the test error stops
# improving - the curve starts going wild and chasing the noise.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(42)
X = np.sort(rng.uniform(0, 1, 40)).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + rng.normal(0, 0.2, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

for degree in [1, 3, 8, 20]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    train_err = mean_squared_error(y_train, model.predict(X_train))
    test_err = mean_squared_error(y_test, model.predict(X_test))
    print(f"Degree {degree}: train MSE {train_err:.4f}, test MSE {test_err:.4f}")
print("The curve starts going wild around degree 8 (test error rises again).")
