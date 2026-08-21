import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2, os

os.makedirs("figures", exist_ok=True)

results = {}
for name in ["knn", "cnn", "rnn"]:
    with open(f"results_{name}.json") as f:
        results[name] = json.load(f)

classes = results["cnn"]["classes"]

# 1. Accuracy comparison bar chart
labels = [results[k]["model"] for k in ["knn", "cnn", "rnn"]]
accs = [results[k]["accuracy"] * 100 for k in ["knn", "cnn", "rnn"]]
plt.figure(figsize=(6, 4.5))
bars = plt.bar(labels, accs, color=["#6c8ebf", "#82b366", "#d79b00"])
plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison: Emotion Recognition Accuracy")
plt.ylim(0, 100)
for b, a in zip(bars, accs):
    plt.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/accuracy_comparison.png", dpi=150)
plt.close()

# 2. Macro F1 comparison
f1s = [results[k]["classification_report"]["macro avg"]["f1-score"] * 100 for k in ["knn", "cnn", "rnn"]]
plt.figure(figsize=(6, 4.5))
bars = plt.bar(labels, f1s, color=["#6c8ebf", "#82b366", "#d79b00"])
plt.ylabel("Macro F1-score (%)")
plt.title("Model Comparison: Macro F1-score")
plt.ylim(0, 100)
for b, a in zip(bars, f1s):
    plt.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/f1_comparison.png", dpi=150)
plt.close()

# 3. Confusion matrices
for k in ["knn", "cnn", "rnn"]:
    cm = np.array(results[k]["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Proportion")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {results[k]['model']}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_{k}.png", dpi=150)
    plt.close()

# 4. Training curves for CNN and RNN
for k in ["cnn", "rnn"]:
    hist = results[k]["history"]
    epochs = range(1, len(hist["accuracy"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, hist["accuracy"], label="train")
    axes[0].plot(epochs, hist["val_accuracy"], label="val")
    axes[0].set_title(f"{results[k]['model']} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, hist["loss"], label="train")
    axes[1].plot(epochs, hist["val_loss"], label="val")
    axes[1].set_title(f"{results[k]['model']} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"figures/training_curve_{k}.png", dpi=150)
    plt.close()

# 5. Haar cascade face-detection screenshot (before/after) for section 5.1
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
sample_files = [
    "facial_expressions-master/images/AJ_Cook_0001.jpg",
    "facial_expressions-master/images/Aaron_Eckhart_0001.jpg",
    "facial_expressions-master/images/Aaron_Guiel_0001.jpg",
]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for col, path in enumerate(sample_files):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[0, col].imshow(img_rgb)
    axes[0, col].set_title("Original")
    axes[0, col].axis("off")
    img_box = img_rgb.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 3)
    axes[1, col].imshow(img_box)
    axes[1, col].set_title("Haar Cascade Detection")
    axes[1, col].axis("off")
plt.tight_layout()
plt.savefig("figures/haar_cascade_demo.png", dpi=150)
plt.close()

# 6. Sample preprocessed (48x48 grayscale) training images grid, per class
X_train = np.load("arrays/X_train.npy")
y_train = np.load("arrays/y_train.npy")
fig, axes = plt.subplots(1, len(classes), figsize=(2.2 * len(classes), 2.6))
for i, cls in enumerate(classes):
    idx = np.where(y_train == i)[0][0]
    axes[i].imshow(X_train[idx], cmap="gray")
    axes[i].set_title(cls)
    axes[i].axis("off")
plt.suptitle("Preprocessed 48x48 face crops per class")
plt.tight_layout()
plt.savefig("figures/sample_preprocessed.png", dpi=150)
plt.close()

print("Saved all figures to figures/")
print(os.listdir("figures"))
