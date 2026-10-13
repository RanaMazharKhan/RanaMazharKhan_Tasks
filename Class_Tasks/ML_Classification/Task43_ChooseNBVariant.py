# Practice Q6 - Which Naive Bayes variant for: (a) exam scores, (b) word counts, (c) button clicks?
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# (a) exam scores are continuous numbers -> GaussianNB
rng = np.random.default_rng(0)
scores_a = rng.normal(70, 10, (60, 3))
scores_b = rng.normal(55, 12, (60, 3))
X_scores = np.vstack([scores_a, scores_b])
y_scores = np.array([1] * 60 + [0] * 60)
gauss = GaussianNB().fit(X_scores, y_scores)
print("(a) exam scores -> GaussianNB, train accuracy:", round(gauss.score(X_scores, y_scores), 3))

# (b) word counts in articles -> MultinomialNB
counts = rng.poisson([[3, 1, 0, 2]] * 40 + [[0, 2, 4, 1]] * 40)
y_words = np.array([0] * 40 + [1] * 40)
multi = MultinomialNB().fit(counts, y_words)
print("(b) word counts  -> MultinomialNB, train accuracy:", round(multi.score(counts, y_words), 3))

# (c) clicked / not clicked on 50 buttons (yes/no features) -> BernoulliNB
clicks = (rng.random((80, 50)) < 0.3).astype(int)
y_clicks = (clicks[:, :5].sum(axis=1) > 2).astype(int)
bern = BernoulliNB().fit(clicks, y_clicks)
print("(c) button clicks -> BernoulliNB, train accuracy:", round(bern.score(clicks, y_clicks), 3))

# GaussianNB for continuous values, MultinomialNB for counts, BernoulliNB for
# binary yes/no features.
