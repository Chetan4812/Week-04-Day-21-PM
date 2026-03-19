import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

np.random.seed(42)

# --- Regression ---
X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_tr, y_tr)
y_pred = reg.predict(X_te)

print(f"Linear Regression")
print(f"  Coefficient : {reg.coef_[0]:.4f}")
print(f"  Intercept   : {reg.intercept_:.4f}")
print(f"  MSE         : {mean_squared_error(y_te, y_pred):.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_te, y_te, color='steelblue', alpha=0.7, label='Actual')
plt.plot(X_te, y_pred, color='crimson', linewidth=2, label='Predicted Line')
plt.title('Linear Regression — Actual vs Predicted')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.tight_layout()
plt.savefig('regression_fit.png', dpi=150)
plt.show()
# Interpretation: The red line captures the linear trend; scatter around it reflects noise.

# --- Classification ---
X_clf, y_clf = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42
)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_tr2, y_tr2)
y_pred2 = clf.predict(X_te2)

print(f"\nLogistic Regression")
print(f"  Accuracy : {accuracy_score(y_te2, y_pred2) * 100:.2f}%")

# Decision boundary
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 5))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
sc = plt.scatter(X_te2[:, 0], X_te2[:, 1], c=y_te2, cmap='RdBu', edgecolors='k', s=60)
plt.colorbar(sc, label='Class')
plt.title('Logistic Regression — Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.savefig('classification_boundary.png', dpi=150)
plt.show()
# Interpretation: Shaded regions show class boundaries; Blue = Class 0, Red = Class 1.
