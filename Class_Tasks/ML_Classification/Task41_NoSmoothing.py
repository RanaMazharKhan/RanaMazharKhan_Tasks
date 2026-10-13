# Practice Q4 - Remove smoothing (alpha=1e-10) and predict a message with an unseen word.
import warnings
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    ("win a free prize now", "spam"), ("free money click now", "spam"),
    ("claim your free gift card", "spam"), ("urgent you won cash prize", "spam"),
    ("limited offer buy now cheap", "spam"), ("free entry win cash now", "spam"),
    ("congratulations winner claim prize", "spam"), ("cheap loans click here now", "spam"),
    ("are we meeting for lunch today", "ham"), ("please send the report by friday", "ham"),
    ("can you call me after the class", "ham"), ("the meeting is moved to monday", "ham"),
    ("happy birthday see you at dinner", "ham"), ("i will send the notes tonight", "ham"),
    ("lunch at noon in the cafeteria", "ham"), ("thanks for the help with homework", "ham"),
]
texts = [m[0] for m in messages]
labels = [m[1] for m in messages]

cv = CountVectorizer()
X_counts = cv.fit_transform(texts)

weird_msg = ["buy cryptocurrency now"]  # 'cryptocurrency' never seen in training
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for alpha in [1.0, 1e-10]:
        mnb = MultinomialNB(alpha=alpha)
        mnb.fit(X_counts, labels)
        proba = mnb.predict_proba(cv.transform(weird_msg))[0]
        print(f"alpha = {alpha}: P(ham) = {proba[0]:.3f}, P(spam) = {proba[1]:.3f}")

# Without real smoothing the probabilities slam to 0 or 1 (over-confident),
# and with alpha = 0 exactly an unseen word would zero out the whole class
# score - that is the zero-probability problem smoothing exists to fix.
