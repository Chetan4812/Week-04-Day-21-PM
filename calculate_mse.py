import numpy as np

def calculate_mse(y_true, y_pred):
    # Formula: MSE = (1/n) * Σ(y_true - y_pred)²
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    return np.sum((y_true - y_pred) ** 2) / n


# --- Test Cases ---
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

mse = calculate_mse(y_true, y_pred)
print(f"y_true : {y_true}")
print(f"y_pred : {y_pred}")
print(f"MSE    : {mse:.4f}")
# Output: MSE = 0.3750

# Perfect prediction → MSE = 0
print(f"\nPerfect prediction MSE : {calculate_mse([1, 2, 3], [1, 2, 3]):.4f}")
