# Practice Q5 - Try 30 copies of each feature. What happens to the average confidence?
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer(as_frame=True)
Xc, yc = cancer.data, cancer.target
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.25, random_state=0, stratify=yc)

rows = []
for k in [1, 3, 10, 30]:
    Xk_tr = np.hstack([Xc_tr.values] * k)
    Xk_te = np.hstack([Xc_te.values] * k)
    m = GaussianNB().fit(Xk_tr, yc_tr)
    p = m.predict_proba(Xk_te)
    rows.append({"copies": k,
                 "accuracy": round(accuracy_score(yc_te, m.predict(Xk_te)), 3),
                 "avg_confidence": round(p.max(axis=1).mean(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# Accuracy barely moves, but the average confidence keeps climbing toward 1.
# Copying features adds no new information - Naive Bayes just counts the same
# evidence again and again, so it becomes over-confident.
