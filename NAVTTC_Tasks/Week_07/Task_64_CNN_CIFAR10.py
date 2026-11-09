# Task 64 - Deep Learning - Perform CNN on CIFAR-10 Dataset
# Problem: CIFAR-10 has 60000 colour images of size 32x32 in 10
# classes. Show the data preparation steps (shapes, normalizing,
# one-hot labels) with numpy on a small sample batch, the same
# steps the real CNN training code below uses.

import numpy as np

classes = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
print("Number of classes:", len(classes))

# one batch of 4 sample images with random pixel values 0-255
np.random.seed(3)
x_sample = np.random.randint(0, 256, size=(4, 32, 32, 3))
y_sample = np.array([0, 3, 5, 9])
print("Image batch shape:", x_sample.shape, "(images, height, width, channels)")

# step 1: normalize pixels from 0-255 to 0-1
x_norm = x_sample / 255.0
print("After normalizing, pixel range:", x_norm.min(), "to", round(float(x_norm.max()), 2))

# step 2: one-hot encode the labels
y_onehot = np.zeros((len(y_sample), 10))
y_onehot[np.arange(len(y_sample)), y_sample] = 1
print("Labels:", [classes[i] for i in y_sample])
print("One-hot label of first image:", y_onehot[0])

# Real training on CIFAR-10 with Keras (pip install tensorflow):
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.utils import to_categorical
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)
#
# model = Sequential([
#     Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation="relu"),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(64, activation="relu"),
#     Dense(10, activation="softmax")
# ])
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
# loss, acc = model.evaluate(x_test, y_test)
# print("Test accuracy:", acc)
