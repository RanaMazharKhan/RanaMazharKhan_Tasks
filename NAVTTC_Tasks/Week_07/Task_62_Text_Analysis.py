# Task 62 - Deep Learning - Text Analysis
# Problem: Analyse a sample review text: count words and
# sentences, find the longest word, count the most repeated
# words, and give a simple sentiment result (positive/negative)
# using small word lists.

import re
from collections import Counter

text = """The course was very good and the teacher was excellent.
I learned many new things and the practice tasks were helpful.
The pace was slow on some days but overall the experience was great and useful."""

words = re.findall(r"[a-z]+", text.lower())
sentences = [s for s in re.split(r"[.!?]", text) if s.strip()]

print("Total words:", len(words))
print("Total sentences:", len(sentences))
print("Average words per sentence:", round(len(words) / len(sentences), 1))
print("Longest word:", max(words, key=len))
print("Most repeated words:", Counter(words).most_common(3))

positive = {"good", "excellent", "helpful", "great", "useful", "new"}
negative = {"slow", "bad", "poor", "boring", "hard"}

pos_count = sum(1 for w in words if w in positive)
neg_count = sum(1 for w in words if w in negative)
print("\nPositive words found:", pos_count)
print("Negative words found:", neg_count)
if pos_count > neg_count:
    print("Sentiment: Positive")
elif neg_count > pos_count:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
