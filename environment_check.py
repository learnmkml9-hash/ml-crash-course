import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("Python executable:", sys.executable)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# Tiny synthetic regression example
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 1))
y = 3.0 * X[:, 0] + 2.0 + 0.5 * rng.normal(size=100)

model = LinearRegression()
model.fit(X, y)

print("Estimated slope:", model.coef_[0])
print("Estimated intercept:", model.intercept_)

plt.scatter(X[:, 0], y, label="data")
plt.plot(X[:, 0], model.predict(X), label="linear fit")
plt.legend()
plt.title("Environment check: linear regression")
plt.show()