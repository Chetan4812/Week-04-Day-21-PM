import numpy as np
from sklearn.datasets import make_regression, make_classification

# --- Regression Dataset (continuous target) ---
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
# X_reg shape: (200, 1), y_reg shape: (200,) — continuous float values

# --- Classification Dataset (binary target) ---
X_clf, y_clf = make_classification(
    n_samples=200, n_features=2, n_informative=2,
    n_redundant=0, random_state=42
)
# X_clf shape: (200, 2), y_clf shape: (200,) — values are 0 or 1

print("Regression Dataset")
print(f"  X shape : {X_reg.shape}")
print(f"  y range : [{y_reg.min():.2f}, {y_reg.max():.2f}]  (continuous)")

print("\nClassification Dataset")
print(f"  X shape  : {X_clf.shape}")
print(f"  Classes  : {np.unique(y_clf)}  (binary)")
