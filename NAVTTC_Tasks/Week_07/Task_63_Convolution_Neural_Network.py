# Task 63 - Deep Learning - Demonstrate Convolution Neural Network
# Problem: Demonstrate the main steps of a CNN using numpy only:
# convolve a small image with an edge-detection kernel, apply
# relu, then do max pooling. Print every step.

import numpy as np

# a small 6x6 "image": left half dark, right half bright
image = np.array([[0, 0, 0, 10, 10, 10],
                  [0, 0, 0, 10, 10, 10],
                  [0, 0, 0, 10, 10, 10],
                  [0, 0, 0, 10, 10, 10],
                  [0, 0, 0, 10, 10, 10],
                  [0, 0, 0, 10, 10, 10]])
print("Image:")
print(image)

# vertical edge detection kernel
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])

def convolve(image, kernel):
    kh, kw = kernel.shape
    ih, iw = image.shape
    out = np.zeros((ih - kh + 1, iw - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            part = image[i:i + kh, j:j + kw]
            out[i, j] = np.sum(part * kernel)
    return out

feature_map = convolve(image, kernel)
print("\nFeature map after convolution (edge is detected in the middle):")
print(feature_map)

relu_map = np.maximum(0, feature_map)
print("\nAfter relu:")
print(relu_map)

def max_pool(data, size=2):
    h, w = data.shape
    out = np.zeros((h // size, w // size))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            part = data[i * size:(i + 1) * size, j * size:(j + 1) * size]
            out[i, j] = np.max(part)
    return out

pooled = max_pool(relu_map)
print("\nAfter 2x2 max pooling (size is reduced):")
print(pooled)
print("Note: CNN = convolution + relu + pooling, repeated, then a classifier.")
