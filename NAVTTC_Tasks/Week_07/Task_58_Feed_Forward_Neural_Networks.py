# Task 58 - Deep Learning - Feed Forward Neural Network
# Problem: Show one forward pass of a feed forward network. Input
# goes input layer -> hidden layer (relu) -> output layer (softmax).
# Print the output of every layer for one sample input.

import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

# one sample input with 3 values
x = np.array([0.5, -0.2, 0.8])
print("Input:", x)

# hidden layer: 3 inputs -> 4 neurons
W1 = np.array([[0.2, -0.3, 0.5, 0.1],
               [0.4, 0.1, -0.2, 0.3],
               [-0.1, 0.6, 0.2, -0.4]])
b1 = np.array([0.1, 0.0, -0.1, 0.2])
hidden = relu(np.dot(x, W1) + b1)
print("Hidden layer (relu):", np.round(hidden, 3))

# output layer: 4 neurons -> 3 classes
W2 = np.array([[0.3, -0.2, 0.1],
               [0.1, 0.5, -0.3],
               [-0.4, 0.2, 0.6],
               [0.2, 0.1, -0.1]])
b2 = np.array([0.0, 0.1, -0.1])
output = softmax(np.dot(hidden, W2) + b2)
print("Output layer (softmax):", np.round(output, 3))
print("Predicted class:", int(np.argmax(output)))
print("Note: information only moves forward, there is no loop back.")
