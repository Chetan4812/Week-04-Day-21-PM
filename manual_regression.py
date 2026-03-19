import numpy as np

# Sample dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 4.0, 5.8, 8.1, 10.0], dtype=float)

# Compute slope (m) and intercept (b) using least squares formula
# m = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
x_mean = np.mean(X)
y_mean = np.mean(y)

m = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
b = y_mean - m * x_mean

# Predict using linear equation: y = mx + b
y_pred = m * X + b

# Compute Mean Squared Error: MSE = (1/n) * Σ(y_true - y_pred)²
n = len(y)
mse = np.sum((y - y_pred) ** 2) / n

print(f"Slope (m)     : {m:.4f}")
print(f"Intercept (b) : {b:.4f}")
print(f"Equation      : y = {m:.4f}x + {b:.4f}")
print(f"Predictions   : {np.round(y_pred, 2)}")
print(f"MSE           : {mse:.6f}")
