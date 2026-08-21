"""
Data preparation for the Emotion Recognition project.

Dataset: 'facial_expressions' (muxspace) - built from LFW face crops,
labelled into 8 emotion categories. We keep the 6 categories with
enough samples for a meaningful train/test split (neutral, happiness,
surprise, sadness, anger, disgust) and drop fear/contempt (21 and 9
samples respectively - too few to split reliably). This mirrors the
"Data Imbalances" limitation already called out in the project report.

Steps:
  1. Read legend.csv (filename -> emotion label)
  2. Cap over-represented classes (neutral/happiness) via random
     sampling so the CNN/RNN aren't just learning to predict the
     majority class, while keeping the *relative* imbalance realistic.
  3. Run Haar Cascade face detection on each image and crop to the
     detected face (falls back to the full image if no face is found).
  4. Convert to grayscale, resize to 48x48, normalize to [0,1].
  5. Stratified train/test split (80/20), saved as .npy arrays.
"""
import os
import cv2
import numpy as np
import pandas as pd

RNG_SEED = 42
IMG_SIZE = 48
DATA_DIR = "facial_expressions-master"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LEGEND_PATH = os.path.join(DATA_DIR, "data", "legend.csv")
OUT_DIR = "arrays"
os.makedirs(OUT_DIR, exist_ok=True)

KEEP_CLASSES = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust"]
CAP_PER_CLASS = 700  # downsample majority classes for faster, less biased training

rng = np.random.default_rng(RNG_SEED)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def load_and_crop_face(path):
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # take the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        crop = gray[y:y + h, x:x + w]
    else:
        crop = gray  # fall back to full frame if Haar cascade finds nothing
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return crop


def main():
    legend = pd.read_csv(LEGEND_PATH)
    legend["emotion"] = legend["emotion"].str.lower()
    legend = legend[legend["emotion"].isin(KEEP_CLASSES)].reset_index(drop=True)

    print("Raw class counts (after dropping fear/contempt):")
    print(legend["emotion"].value_counts())

    # cap majority classes
    parts = []
    for cls, grp in legend.groupby("emotion"):
        if len(grp) > CAP_PER_CLASS:
            grp = grp.sample(n=CAP_PER_CLASS, random_state=RNG_SEED)
        parts.append(grp)
    legend = pd.concat(parts).reset_index(drop=True)

    print("\nClass counts after capping majority classes at", CAP_PER_CLASS)
    print(legend["emotion"].value_counts())

    faces_no_detect = 0
    X, y, kept_files = [], [], []
    for i, row in legend.iterrows():
        path = os.path.join(IMAGES_DIR, row["image"])
        crop = load_and_crop_face(path)
        if crop is None:
            continue
        X.append(crop)
        y.append(row["emotion"])
        kept_files.append(row["image"])
        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(legend)}")

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y)

    classes = sorted(KEEP_CLASSES)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=RNG_SEED, stratify=y_idx
    )

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

    print("\nFinal dataset:", X.shape, "images")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Classes:", classes)


if __name__ == "__main__":
    main()
