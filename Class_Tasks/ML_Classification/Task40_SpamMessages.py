# Practice Q3 - Add three of your own spam and ham messages. Do the new_msgs predictions change?
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

messages = [
    ("win a free prize now", "spam"), ("free money click now", "spam"),
    ("claim your free gift card", "spam"), ("urgent you won cash prize", "spam"),
    ("limited offer buy now cheap", "spam"), ("free entry win cash now", "spam"),
    ("congratulations winner claim prize", "spam"), ("cheap loans click here now", "spam"),
    # my own spam messages
    ("you won a free iphone click here", "spam"),
    ("double your money fast guaranteed", "spam"),
    ("act now limited lottery winner", "spam"),
    ("are we meeting for lunch today", "ham"), ("please send the report by friday", "ham"),
    ("can you call me after the class", "ham"), ("the meeting is moved to monday", "ham"),
    ("happy birthday see you at dinner", "ham"), ("i will send the notes tonight", "ham"),
    ("lunch at noon in the cafeteria", "ham"), ("thanks for the help with homework", "ham"),
    # my own ham messages
    ("mom called about dinner on sunday", "ham"),
    ("the football match starts at nine", "ham"),
    ("remember to buy milk and bread", "ham"),
]
texts = [m[0] for m in messages]
labels = [m[1] for m in messages]

cv = CountVectorizer()
X_counts = cv.fit_transform(texts)
mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_counts, labels)

new_msgs = ["free cash prize now", "see you at lunch", "send me the report", "win a free gift"]
pred = mnb.predict(cv.transform(new_msgs))
proba = mnb.predict_proba(cv.transform(new_msgs))

result = pd.DataFrame({"message": new_msgs, "prediction": pred,
                       "P(ham)": proba[:, 0].round(3), "P(spam)": proba[:, 1].round(3)})
print(result.to_string(index=False))

# The clear spam and clear ham messages still get the same labels, but the
# probabilities shift a little because the extra messages changed the word
# counts the model learned from.
