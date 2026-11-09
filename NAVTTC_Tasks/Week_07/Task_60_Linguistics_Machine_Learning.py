# Task 60 - Deep Learning - Linguistics using Machine Learning
# Problem: Do basic NLP on a small paragraph in pure python:
# tokenize the text, remove stop words, count word frequencies
# and print the most common words and the bigrams.

import re
from collections import Counter

text = """Machine learning is a part of artificial intelligence.
Natural language processing helps machines understand human language.
Machines learn language from data and text."""

stopwords = {"is", "a", "of", "and", "from"}

words = re.findall(r"[a-z]+", text.lower())
print("Total words:", len(words))

tokens = [w for w in words if w not in stopwords]
print("Words after stop word removal:", len(tokens))

freq = Counter(tokens)
print("\nMost common words:")
for word, count in freq.most_common(5):
    print(word, ":", count)

bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
print("\nMost common bigrams:")
for pair, count in Counter(bigrams).most_common(3):
    print(pair, ":", count)
