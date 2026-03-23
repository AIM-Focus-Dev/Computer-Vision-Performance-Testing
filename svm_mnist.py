import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Step 1: Load MNIST Dataset
mnist = datasets.fetch_openml('mnist_784', version=1)  # Fetches MNIST
X, y = mnist.data, mnist.target.astype(int)

# Normalise the data
X = X / 255.0  # Scale pixel values to [0, 1]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to binary (e.g., 0 vs. all) for binary ROC curve
y_train_binary = (y_train == 0).astype(int)
y_test_binary = (y_test == 0).astype(int)

# Step 2: Train the SVM Model
svc = SVC(kernel='rbf', C=1, gamma=0.01, probability=True, random_state=42)
svc.fit(X_train, y_train_binary)

# Step 3: Predictions and Metrics
y_pred = svc.predict(X_test)
y_pred_prob = svc.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve

# Calculate metrics
accuracy = accuracy_score(y_test_binary, y_pred)
precision = precision_score(y_test_binary, y_pred)
recall = recall_score(y_test_binary, y_pred)

print("SVM Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred))

# Step 4: ROC Curve
fpr, tpr, _ = roc_curve(y_test_binary, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.title("ROC Curve - SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

