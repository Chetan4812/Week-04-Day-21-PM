import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Regression side: continuous output along a fitted line ---
x = np.linspace(0, 10, 100)
y = 2.5 * x + np.random.default_rng(42).normal(0, 2, 100)
axes[0].scatter(x, y, alpha=0.5, color='steelblue', s=20)
axes[0].plot(x, 2.5 * x, 'r-', linewidth=2, label='Fitted line')
axes[0].set_title('Regression\n(Continuous Output)', fontweight='bold')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y  (continuous)')
axes[0].legend()

# --- Classification side: sigmoid curve with threshold boundary ---
x2 = np.linspace(-5, 5, 300)
sigmoid = 1 / (1 + np.exp(-x2))
axes[1].plot(x2, sigmoid, 'purple', linewidth=2, label='Sigmoid')
axes[1].axhline(0.5, linestyle='--', color='gray', label='Threshold = 0.5')
axes[1].fill_between(x2, 0, sigmoid, where=(sigmoid >= 0.5), alpha=0.2,
                     color='green', label='Class 1')
axes[1].fill_between(x2, 0, sigmoid, where=(sigmoid < 0.5), alpha=0.2,
                     color='red', label='Class 0')
axes[1].set_title('Classification\n(Discrete Output)', fontweight='bold')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Probability')
axes[1].legend()

plt.suptitle('Regression vs Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_vs_classification.png', dpi=150)
plt.show()

# Key differences:
# | Aspect          | Regression         | Classification         |
# |-----------------|--------------------|------------------------|
# | Output          | Continuous float   | Discrete class label   |
# | Use case        | Price, temp, sales | Spam, disease, fraud   |
# | Evaluation      | MSE, RMSE, R²      | Accuracy, F1, AUC-ROC  |
