# Homework 2 - Plot the OOB score against n_estimators from 10 to 300.
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

X, y = load_wine(return_X_y=True)

n_values = list(range(10, 301, 20))
oob_scores = []
for n in n_values:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    oob_scores.append(rf.oob_score_)

plt.figure(figsize=(8, 4.5))
plt.plot(n_values, oob_scores, "o-")
plt.xlabel("n_estimators")
plt.ylabel("OOB score")
plt.title("OOB Score vs Number of Trees")
plt.grid(alpha=0.3)
plt.show()

# The OOB score rises fast at first and then flattens - more trees stop
# helping after a point, but they do not make things worse either.
