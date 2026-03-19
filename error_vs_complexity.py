import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.sort(np.random.uniform(0, 1, 40))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, 40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

degree_range = list(range(1, 13))
train_errors, test_errors = [], []

for d in degree_range:
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train.reshape(-1, 1), y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train.reshape(-1, 1))))
    test_errors.append(mean_squared_error(y_test,  model.predict(X_test.reshape(-1, 1))))

best_degree = degree_range[np.argmin(test_errors)]

plt.figure(figsize=(9, 5))
plt.plot(degree_range, train_errors, 'o-', color='steelblue', linewidth=2, label='Training Error')
plt.plot(degree_range, test_errors,  's-', color='crimson',   linewidth=2, label='Test Error')

# Annotate zones
plt.axvspan(1,   2.5, alpha=0.08, color='red',    label='Underfitting zone')
plt.axvspan(2.5, 5.5, alpha=0.08, color='green',  label='Optimal zone')
plt.axvspan(5.5, 12,  alpha=0.08, color='orange', label='Overfitting zone')
plt.axvline(best_degree, color='green', linestyle='--', linewidth=1.5,
            label=f'Best degree = {best_degree}')

plt.title('Training Error vs Test Error vs Model Complexity', fontweight='bold')
plt.xlabel('Polynomial Degree (Model Complexity →)')
plt.ylabel('Mean Squared Error')
plt.ylim(bottom=0)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('error_vs_complexity.png', dpi=150)
plt.show()

print(f"Optimal polynomial degree: {best_degree}")
