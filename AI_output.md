### Regression vs Classification and Bias–Variance Tradeoff

#### 1. Regression vs Classification

**Regression** predicts a continuous numeric value — the model learns to map inputs to real-valued outputs using a fitting function (e.g., a line or curve).

**Classification** predicts a discrete class label — the model learns a decision boundary that separates input space into regions for each class.

> *Key rule: if the target is a number on a continuous scale → Regression. If it's a category/label → Classification.*

```python
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Regression
X_r, y_r = make_regression(n_samples=100, n_features=1, noise=10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.2)
reg = LinearRegression().fit(X_tr, y_tr)
print(f"Regression MSE: {mean_squared_error(y_te, reg.predict(X_te)):.2f}")

# Classification
X_c, y_c = make_classification(n_samples=100, n_features=2, n_informative=2,
                                n_redundant=0, random_state=0)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_c, y_c, test_size=0.2)
clf = LogisticRegression().fit(X_tr2, y_tr2)
print(f"Classification Accuracy: {accuracy_score(y_te2, clf.predict(X_te2)) * 100:.1f}%")
```

---

#### 2. Bias–Variance Tradeoff

Bias and variance are two sources of prediction error that trade off against each other as model complexity changes.

**Bias** — error from over-simplified assumptions. A high-bias model underfits: it cannot capture the underlying pattern and performs poorly on both train and test data.

**Variance** — error from over-sensitivity to training data. A high-variance model overfits: it memorises noise and performs well on train but poorly on test data.

**Total Error = Bias² + Variance + Irreducible Noise**

The optimal model minimises the sum of bias² and variance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.sort(np.random.uniform(0, 1, 50))
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.3, 50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for degree in [1, 3, 10]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train.reshape(-1, 1), y_train)
    tr = mean_squared_error(y_train, model.predict(X_train.reshape(-1, 1)))
    te = mean_squared_error(y_test,  model.predict(X_test.reshape(-1, 1)))
    status = {1: "Underfitting", 3: "Optimal", 10: "Overfitting"}[degree]
    print(f"Degree {degree:2d} | Train MSE: {tr:.3f} | Test MSE: {te:.3f} | {status}")
```

### Key Differences

| Concept       | Bias                         | Variance                         |
| :------------ | :--------------------------- | :------------------------------- |
| **Cause**     | Model too simple             | Model too complex                |
| **Symptom**   | Underfitting                 | Overfitting                      |
| **Train Err** | High                         | Very low                         |
| **Test Err**  | High                         | High                             |
| **Fix**       | Increase complexity / features | Regularisation, more data, pruning |
