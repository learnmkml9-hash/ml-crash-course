# Lesson 1, Part 2: NumPy arrays and shapes

import numpy as np


# -----------------------------
# 1. Scalars, vectors, matrices
# -----------------------------

scalar = 3.0

vector = np.array([1.0, 2.0, 3.0])

matrix = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])

print("Scalar:")
print(scalar)

print("\nVector:")
print(vector)
print("Vector shape:", vector.shape)

print("\nMatrix:")
print(matrix)
print("Matrix shape:", matrix.shape)


# -----------------------------
# 2. Vector operations
# -----------------------------

x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -1.0, 2.0])

dot_product = np.dot(w, x)

print("\nDot product:")
print(dot_product)


# -----------------------------
# 3. Linear model for one input vector
# -----------------------------

bias = 0.1
prediction = np.dot(w, x) + bias

print("\nLinear prediction for one input:")
print(prediction)


# -----------------------------
# 4. Batch of input vectors (random dataset)
# -----------------------------

# Create a reproducible random dataset: n samples, d features
np.random.seed(42)
n = 100
d = 3
X = np.random.randn(n, d)

# True data-generating parameters (used to create targets)
true_w = np.array([0.5, -1.0, 2.0])
true_b = 0.1

print("\nInput matrix X (first 5 rows):")
print(X[:5])
print("X shape:", X.shape)

print("\nWeight vector w (model parameters):")
print(w)
print("w shape:", w.shape)


# -----------------------------
# 5. Predictions for a batch
# -----------------------------

predictions = X @ w + bias

print("\nBatch predictions:")
print(predictions)
print("Predictions shape:", predictions.shape)


# -----------------------------
# 6. Targets and errors
# -----------------------------

# Generate targets from the true model with additive Gaussian noise
noise = 0.1 * np.random.randn(n)
y = X @ true_w + true_b + noise

errors = predictions - y

print("\nTargets (first 5):")
print(y[:5])
print("Targets shape:", y.shape)

print("\nErrors (first 5):")
print(errors[:5])
print("Errors shape:", errors.shape)


# -----------------------------
# 7. Mean squared error
# -----------------------------

mse = np.mean(errors ** 2)

print("\nMean squared error:")
print(mse)