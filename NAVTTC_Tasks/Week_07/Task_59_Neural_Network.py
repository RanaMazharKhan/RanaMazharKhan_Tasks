# Task 59 - Deep Learning - Neural Network (Loss Functions)
# Problem: The linked guide is about loss functions. Implement 7
# common loss functions in numpy and print the loss value of each
# for the same sample predictions, so they can be compared.

import numpy as np

y_true = np.array([1.0, 0.0, 1.0, 1.0])
y_pred = np.array([0.9, 0.2, 0.8, 0.6])
y_true_multi = np.array([0, 0, 1, 0])
y_pred_multi = np.array([0.1, 0.2, 0.6, 0.1])

def mse(y, p):
    return np.mean((y - p) ** 2)

def mae(y, p):
    return np.mean(np.abs(y - p))

def huber(y, p, delta=1.0):
    error = np.abs(y - p)
    loss = np.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return np.mean(loss)

def binary_cross_entropy(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def categorical_cross_entropy(y_onehot, p):
    p = np.clip(p, 1e-9, 1)
    return -np.sum(y_onehot * np.log(p))

def hinge(y, p):
    y_sign = np.where(y == 1, 1, -1)
    return np.mean(np.maximum(0, 1 - y_sign * p))

def kl_divergence(y, p):
    p = np.clip(p, 1e-9, 1)
    y = np.clip(y, 1e-9, 1)
    return np.sum(y * np.log(y / p))

print("Mean Squared Error:      ", round(mse(y_true, y_pred), 4))
print("Mean Absolute Error:     ", round(mae(y_true, y_pred), 4))
print("Huber Loss:              ", round(huber(y_true, y_pred), 4))
print("Binary Cross Entropy:    ", round(binary_cross_entropy(y_true, y_pred), 4))
print("Categorical Cross Entropy:", round(categorical_cross_entropy(y_true_multi, y_pred_multi), 4))
print("Hinge Loss:              ", round(hinge(y_true, y_pred), 4))
print("KL Divergence:           ", round(kl_divergence(y_true_multi + 0.1, y_pred_multi), 4))
print("Note: lower loss means predictions are closer to the true values.")
