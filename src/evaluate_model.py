import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Load model + scaler
with open("action_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X = scaler.transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Aggressive", "Fight"],
            yticklabels=["Normal", "Aggressive", "Fight"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Class accuracy graph
class_accuracy = cm.diagonal() / cm.sum(axis=1)

plt.figure()
plt.bar(["Normal", "Aggressive", "Fight"], class_accuracy)
plt.title("Class-wise Accuracy")
plt.ylabel("Accuracy")
plt.savefig("class_accuracy.png")
plt.close()

print("\nGraphs saved successfully!")
