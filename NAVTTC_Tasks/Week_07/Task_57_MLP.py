# Task 57 - Deep Learning - MLP (Multi Layer Perceptron)
# Problem: Train an MLP from scratch (numpy) on a small two-class
# dataset of 100 points. Print the training loss and the accuracy.

import numpy as np

np.random.seed(2)

# two groups of points: class 0 near (0, 0), class 1 near (2, 2)
X0 = np.random.randn(50, 2) * 0.6 + [0, 0]
X1 = np.random.randn(50, 2) * 0.6 + [2, 2]
X = np.vstack([X0, X1])
y = np.array([0] * 50 + [1] * 50).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# MLP: input(2) -> hidden(5) -> output(1)
W1 = np.random.randn(2, 5) * 0.5
b1 = np.zeros((1, 5))
W2 = np.random.randn(5, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.1
for epoch in range(1000):
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    d_output = (output - y) * output * (1 - output)
    d_hidden = np.dot(d_output, W2.T) * hidden * (1 - hidden)

    W2 -= lr * np.dot(hidden.T, d_output) / len(X)
    b2 -= lr * np.mean(d_output, axis=0, keepdims=True)
    W1 -= lr * np.dot(X.T, d_hidden) / len(X)
    b1 -= lr * np.mean(d_hidden, axis=0, keepdims=True)

    if (epoch + 1) % 250 == 0:
        loss = np.mean((output - y) ** 2)
        print("Epoch", epoch + 1, "loss:", round(float(loss), 4))

predictions = (output >= 0.5).astype(int)
accuracy = np.mean(predictions == y)
print("Accuracy:", round(float(accuracy) * 100, 1), "%")
print("First 5 predictions:", predictions[:5].flatten())
print("First 5 actual:     ", y[:5].flatten())
