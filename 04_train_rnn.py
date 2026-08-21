"""
Recurrent Neural Network (LSTM) for emotion recognition.
Each 48x48 image is treated as a sequence of 48 rows, each row a
48-dim feature vector - a common lightweight way student projects
apply RNNs to static images (as opposed to true video/temporal data).
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

tf.random.set_seed(42)

X_train = np.load("arrays/X_train.npy")  # (N, 48, 48)
X_test = np.load("arrays/X_test.npy")
y_train = np.load("arrays/y_train.npy")
y_test = np.load("arrays/y_test.npy")
classes = open("arrays/classes.txt").read().splitlines()
n_classes = len(classes)

class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

model = models.Sequential([
    layers.Input(shape=(48, 48)),  # 48 timesteps, 48 features (pixel rows)
    layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
    layers.Dropout(0.3),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(n_classes, activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=40,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=2,
)

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}")
report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=classes))

results = {
    "model": "RNN (Bi-LSTM)",
    "accuracy": float(acc),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "classes": classes,
    "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    "epochs_trained": len(history.history["loss"]),
}
with open("results_rnn.json", "w") as f:
    json.dump(results, f, indent=2)
model.save("rnn_model.keras")
print("Saved results_rnn.json and rnn_model.keras")
