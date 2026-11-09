# Task 54 - Machine Learning - Support Vector Machine (SVM)
# Problem: Train a simple linear SVM from scratch (hinge loss +
# gradient descent) on two small groups of 2-D points. Print the
# learned weights, the margin, and predict the class of [3, 3].

import numpy as np

# class -1 points are near (1, 1), class +1 points are near (4, 4)
X = np.array([[1.0, 1.0], [1.5, 2.0], [2.0, 1.0],
              [4.0, 4.0], [4.5, 3.5], [5.0, 4.5]])
y = np.array([-1, -1, -1, 1, 1, 1])

w = np.zeros(2)
b = 0.0
lr = 0.1
reg = 0.01

for epoch in range(100):
    for i in range(len(X)):
        margin_i = y[i] * (np.dot(w, X[i]) + b)
        if margin_i < 1:
            w = w + lr * (y[i] * X[i] - 2 * reg * w)
            b = b + lr * y[i]
        else:
            w = w + lr * (-2 * reg * w)

print("Weights:", np.round(w, 2))
print("Bias:", round(b, 2))
print("Margin width:", round(2 / np.linalg.norm(w), 2))

def predict(x):
    if np.dot(w, x) + b >= 0:
        return 1
    return -1

print("Training predictions:", [predict(x) for x in X])
print("Prediction for [3, 3]:", predict([3, 3]))
print("Prediction for [1, 2]:", predict([1, 2]))

# Same thing with scikit-learn (pip install scikit-learn):
# from sklearn.svm import SVC
# clf = SVC(kernel="linear")
# clf.fit(X, y)
# print(clf.predict([[3, 3]]))
