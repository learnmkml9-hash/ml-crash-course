import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Python executable:")
print(sys.executable)

print("\nPackage versions:")
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

# Generate synthetic data: y = 3x + 2 + noise
rng = np.random.default_rng(0)

X = rng.normal(size=(100, 1))
noise = 0.5 * rng.normal(size=100)
y = 3.0 * X[:, 0] + 2.0 + noise

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

print("\nLearned linear model:")
print("Estimated slope:", model.coef_[0])
print("Estimated intercept:", model.intercept_)

# Plot data and fitted line
plt.scatter(X[:, 0], y, label="data")
plt.scatter(X[:, 0], model.predict(X), label="model prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("First ML environment check")
plt.legend()
plt.show()