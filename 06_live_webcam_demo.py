"""
Live webcam emotion recognition demo.

Opens your webcam, detects faces with the same Haar Cascade used during
training, crops+preprocesses each face exactly like 01_data_prep.py did,
runs it through the trained CNN (cnn_model.keras), and draws a box with
the predicted emotion + confidence on the live video feed.

Run this AFTER 01-05 have been run at least once (needs cnn_model.keras
and arrays/classes.txt, both already included in the project folder /
created by the earlier scripts).

Controls:
  q  -> quit
  s  -> save a snapshot of the current frame to snapshots/
"""
import os
import cv2
import numpy as np
from tensorflow import keras

IMG_SIZE = 48
MODEL_PATH = "cnn_model.keras"
CLASSES_PATH = "arrays/classes.txt"
SNAPSHOT_DIR = "snapshots"

if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Couldn't find {MODEL_PATH} - run 03_train_cnn.py first.")
if not os.path.exists(CLASSES_PATH):
    raise SystemExit(f"Couldn't find {CLASSES_PATH} - run 01_data_prep.py first.")

classes = open(CLASSES_PATH).read().splitlines()
model = keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# emotion -> BGR box color, just for a bit of visual flair
COLORS = {
    "happiness": (0, 200, 0),
    "neutral": (200, 200, 200),
    "surprise": (0, 200, 255),
    "sadness": (255, 100, 0),
    "anger": (0, 0, 255),
    "disgust": (0, 128, 0),
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Is it connected / not used by another app?")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print("Live emotion recognition running. Press 'q' to quit, 's' to save a snapshot.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Failed to read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_crop = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        face_input = face_resized.astype(np.float32) / 255.0
        face_input = face_input[np.newaxis, ..., np.newaxis]  # (1, 48, 48, 1)

        probs = model.predict(face_input, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = classes[idx]
        confidence = probs[idx] * 100

        color = COLORS.get(label, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{label} ({confidence:.0f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Emotion Recognition - press q to quit", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        path = os.path.join(SNAPSHOT_DIR, f"snapshot_{cv2.getTickCount()}.png")
        cv2.imwrite(path, frame)
        print(f"Saved {path}")

cap.release()
cv2.destroyAllWindows()
