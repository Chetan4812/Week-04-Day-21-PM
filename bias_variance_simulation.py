import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# True function: y = sin(2πx) + noise
X = np.sort(np.random.uniform(0, 1, 40))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, 40)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_dense = np.linspace(0, 1, 300)

degrees = [1, 2, 5, 10]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train.reshape(-1, 1), y_train)

    y_tr_pred   = model.predict(X_train.reshape(-1, 1))
    y_te_pred   = model.predict(X_test.reshape(-1, 1))
    y_dense_pred = model.predict(X_dense.reshape(-1, 1))

    tr_err = mean_squared_error(y_train, y_tr_pred)
    te_err = mean_squared_error(y_test, y_te_pred)

    print(f"Degree {degree:2d} → Train MSE: {tr_err:.4f}  |  Test MSE: {te_err:.4f}", end='  ')
    if degree == 1:  print("← Underfitting (high bias)")
    elif degree == 10: print("← Overfitting (high variance)")
    else: print()

    ax = axes[i]
    ax.scatter(X_train, y_train, color='steelblue', s=30, alpha=0.7, label='Train')
    ax.scatter(X_test,  y_test,  color='orange',    s=30, alpha=0.7, label='Test')
    ax.plot(X_dense, np.sin(2 * np.pi * X_dense), 'g--', linewidth=1.5, label='True fn')
    ax.plot(X_dense, y_dense_pred, 'r-', linewidth=2, label=f'Degree {degree}')
    ax.set_ylim(-2.2, 2.2)
    ax.set_title(f'Polynomial Degree = {degree}\nTrain MSE={tr_err:.3f}  |  Test MSE={te_err:.3f}',
                 fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend(fontsize=8)

plt.suptitle('Bias–Variance Tradeoff: Polynomial Fits', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('bias_variance_fits.png', dpi=150)
plt.show()
