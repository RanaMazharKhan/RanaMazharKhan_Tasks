# Homework 2 - Plot train vs test accuracy for min_samples_leaf from 1 to 30.
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

leaf_values = list(range(1, 31))
train_scores, test_scores = [], []
for leaf in leaf_values:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

plt.figure(figsize=(8, 4.5))
plt.plot(leaf_values, train_scores, "o-", label="Train")
plt.plot(leaf_values, test_scores, "s-", label="Test")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Small min_samples_leaf overfits (train ~1.0, test lower); making leaves
# bigger simplifies the tree and the gap between train and test shrinks.
