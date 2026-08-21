"""
K-Nearest Neighbours baseline for emotion recognition.
Features: flattened 48x48 grayscale pixel intensities (2304-d vector).
"""
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train = np.load("arrays/X_train.npy")
X_test = np.load("arrays/X_test.npy")
y_train = np.load("arrays/y_train.npy")
y_test = np.load("arrays/y_test.npy")
classes = open("arrays/classes.txt").read().splitlines()

Xtr = X_train.reshape(len(X_train), -1)
Xte = X_test.reshape(len(X_test), -1)

best_k, best_acc = None, -1
for k in [3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn.fit(Xtr, y_train)
    acc = accuracy_score(y_test, knn.predict(Xte))
    print(f"k={k:>2}  test accuracy={acc:.4f}")
    if acc > best_acc:
        best_acc, best_k = acc, k

print(f"\nBest k = {best_k}, accuracy = {best_acc:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(Xtr, y_train)
y_pred = knn.predict(Xte)

report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=classes))

results = {
    "model": "KNN",
    "best_k": best_k,
    "accuracy": float(best_acc),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "classes": classes,
}
with open("results_knn.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved results_knn.json")
