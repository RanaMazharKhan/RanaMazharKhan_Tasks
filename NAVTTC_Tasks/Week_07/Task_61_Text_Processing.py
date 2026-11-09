# Task 61 - Deep Learning - Text Processing
# Problem: Process a sample text step by step: split it into
# sentences and words, remove stop words, and apply simple
# stemming. Print the result of each step.

import re

text = "Cats are running in the gardens. The dogs were playing and jumping happily."

sentences = re.split(r"[.!?]", text)
sentences = [s.strip() for s in sentences if s.strip()]
print("Sentences:", sentences)

stopwords = {"are", "in", "the", "and", "were"}

def stem(word):
    for suffix in ["ing", "ly", "es", "s"]:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word

for s in sentences:
    words = re.findall(r"[a-z]+", s.lower())
    print("\nWords:", words)
    cleaned = [w for w in words if w not in stopwords]
    print("After stop word removal:", cleaned)
    stemmed = [stem(w) for w in cleaned]
    print("After stemming:", stemmed)

# Same steps with NLTK (pip install nltk):
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# print(sent_tokenize(text))
# tokens = word_tokenize(text)
# clean = [w for w in tokens if w.lower() not in stopwords.words("english")]
# print([PorterStemmer().stem(w) for w in clean])
