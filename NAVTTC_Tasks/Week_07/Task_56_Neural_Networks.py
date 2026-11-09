# Task 56 - Machine Learning - Demonstration of Neural Networks
# Problem: Build a small neural network from scratch (numpy only)
# with one hidden layer and train it on the XOR problem using
# backpropagation. Print the predictions after training.

import numpy as np

np.random.seed(1)

# XOR data: output is 1 only when the two inputs are different
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# weights: input(2) -> hidden(2) -> output(1)
W1 = np.random.randn(2, 2)
b1 = np.zeros((1, 2))
W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

lr = 0.5
for epoch in range(5000):
    # forward pass
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    # backward pass
    d_output = (output - y) * output * (1 - output)
    d_hidden = np.dot(d_output, W2.T) * hidden * (1 - hidden)

    # update weights
    W2 -= lr * np.dot(hidden.T, d_output)
    b2 -= lr * np.sum(d_output, axis=0, keepdims=True)
    W1 -= lr * np.dot(X.T, d_hidden)
    b1 -= lr * np.sum(d_hidden, axis=0, keepdims=True)

    if epoch == 4999:
        error = np.mean((output - y) ** 2)
        print("Final error:", round(float(error), 4))

print("Predictions:")
print(np.round(output, 2))
print("Expected:")
print(y)
