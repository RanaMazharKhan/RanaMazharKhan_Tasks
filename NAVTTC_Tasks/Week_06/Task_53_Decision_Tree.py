# Task 53 - Machine Learning - Decision Tree
# Problem: Build a small decision tree from scratch on fruit data.
# Features are weight and colour (0=green, 1=red, 2=orange), label is
# the fruit name. Print the tree and predict the fruit for [160, 1].

import numpy as np

X = np.array([[140, 1], [130, 1], [150, 2], [170, 2],
              [145, 1], [180, 2], [120, 0], [110, 0]])
y = np.array(["apple", "apple", "orange", "orange",
              "apple", "orange", "lime", "lime"])

def gini(labels):
    total = len(labels)
    impurity = 1.0
    for lab in set(labels):
        p = list(labels).count(lab) / total
        impurity -= p * p
    return impurity

def best_split(X, y):
    best_gain, best_f, best_t = 0, None, None
    base = gini(y)
    n = len(y)
    for f in range(X.shape[1]):
        for t in sorted(set(X[:, f])):
            left = y[X[:, f] <= t]
            right = y[X[:, f] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = base - (len(left) / n) * gini(left) - (len(right) / n) * gini(right)
            if gain > best_gain:
                best_gain, best_f, best_t = gain, f, t
    return best_f, best_t

def most_common(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]

def build_tree(X, y, depth=0):
    if len(set(y)) == 1 or depth == 2:
        print("  " * depth + "-> " + str(most_common(y)))
        return most_common(y)
    f, t = best_split(X, y)
    if f is None:
        print("  " * depth + "-> " + str(most_common(y)))
        return most_common(y)
    print("  " * depth + "if feature " + str(f) + " <= " + str(t) + ":")
    left = build_tree(X[X[:, f] <= t], y[X[:, f] <= t], depth + 1)
    print("  " * depth + "else:")
    right = build_tree(X[X[:, f] > t], y[X[:, f] > t], depth + 1)
    return (f, t, left, right)

def predict(tree, x):
    if not isinstance(tree, tuple):
        return tree
    f, t, left, right = tree
    if x[f] <= t:
        return predict(left, x)
    return predict(right, x)

print("Decision tree:")
tree = build_tree(X, y)
print("Prediction for [160, 1]:", predict(tree, [160, 1]))
print("Prediction for [115, 0]:", predict(tree, [115, 0]))

# Same thing with scikit-learn (pip install scikit-learn):
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# clf.fit(X, y)
# print(clf.predict([[160, 1]]))
