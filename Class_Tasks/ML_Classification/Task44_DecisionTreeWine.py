# Homework 1 - Load the wine dataset and train a decision tree on it.
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

print("Train accuracy:", round(tree.score(X_train, y_train), 3))
print("Test accuracy:", round(tree.score(X_test, y_test), 3))
