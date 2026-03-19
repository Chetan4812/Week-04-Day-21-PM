import numpy as np

# Simulated probability scores (e.g., from a logistic model or heuristic)
prob_scores = np.array([0.1, 0.4, 0.55, 0.7, 0.85, 0.3, 0.65, 0.9, 0.2, 0.48])
y_true      = np.array([0,   0,   1,    1,   1,    0,   1,    1,   0,   0  ])

# Apply threshold to classify: >= 0.5 → Class 1, else Class 0
THRESHOLD = 0.5
y_pred = (prob_scores >= THRESHOLD).astype(int)
# y_pred: [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]

# Compute accuracy manually: correct_predictions / total_predictions
correct = np.sum(y_pred == y_true)
accuracy = correct / len(y_true)

print(f"Predicted : {y_pred}")
print(f"Actual    : {y_true}")
print(f"Correct   : {correct} / {len(y_true)}")
print(f"Accuracy  : {accuracy * 100:.1f}%")
