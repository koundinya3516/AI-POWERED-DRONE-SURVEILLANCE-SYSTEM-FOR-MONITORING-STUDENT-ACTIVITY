import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------- LOAD DATA --------
X = np.load("X.npy")
y = np.load("y.npy")

# -------- SPLIT (for report) --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------- MODEL (with OOB) --------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    oob_score=True,          # ⭐ key addition
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# -------- TRAIN --------
model.fit(X_train, y_train)

# -------- OOB ACCURACY --------
print("\nOOB Accuracy:", model.oob_score_)

# -------- TEST ACCURACY --------
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------- SAVE --------
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel saved as model.pkl")
