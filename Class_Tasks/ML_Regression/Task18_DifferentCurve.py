# Bias-variance try-it-yourself 4: new curve y = 3x + 4*sin(x) + noise.
# R2 looks decent (about 0.7) but the residuals still curve, which means a
# straight line is the wrong shape - judging by R2 alone would fool us.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

rng = np.random.RandomState(42)
X = np.sort(rng.uniform(0, 3, 60)).reshape(-1, 1)
y = 3 * X.ravel() + 4 * np.sin(X.ravel()) + rng.normal(0, 1, 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)
print("R2 of straight line:", round(r2_score(y_test, model.predict(X_test)), 3))

residuals = y_test - model.predict(X_test)
plt.scatter(X_test, residuals)
plt.axhline(0, color="red")
plt.xlabel("x")
plt.ylabel("residual (actual - predicted)")
plt.title("Residuals still curve -> straight line is the wrong shape")
plt.show()
print("The residuals form a wave, so R2 alone hid the real problem.")
