# Emotion Recognition — Comparative Study (Recreated Code + Results)

Recreated implementation for "A Comparative Study of Machine Learning
Algorithms for Emotion Recognition" — rebuilt from the submitted report
since the original code/data/results were lost.

## Dataset
`facial_expressions` by muxspace (built on LFW face crops), 13,690 images
across 8 emotion labels. Download it yourself with:

    git clone https://github.com/muxspace/facial_expressions.git

Place the cloned folder as `facial_expressions-master/` next to these
scripts (or edit `DATA_DIR` in `01_data_prep.py`).

## Pipeline — run in order
1. `01_data_prep.py` — Haar Cascade face crop, grayscale, resize to 48x48,
   class balancing, stratified train/test split → saves `arrays/*.npy`
2. `02_train_knn.py` — K-Nearest Neighbours baseline (raw pixel features)
3. `03_train_cnn.py` — Convolutional Neural Network (Keras/TensorFlow)
4. `04_train_rnn.py` — Bidirectional LSTM, image rows as a sequence
5. `05_make_visuals.py` — accuracy/F1 comparison charts, confusion
   matrices, training curves, Haar Cascade demo screenshots

## Results summary (this run)

| Model          | Test Accuracy | Macro F1 |
|----------------|:---:|:---:|
| KNN (k=7)      | 41.6% | 0.37 |
| CNN            | 58.8% | 0.53 |
| RNN (Bi-LSTM)  | 47.8% | 0.43 |

Full per-class metrics, confusion matrices, and training curves are in
`Results_and_Statistics.docx`.

## Requirements
```
pip install numpy pandas scikit-learn opencv-python tensorflow-cpu matplotlib
```

## Notes / honesty about limitations
- Two of the original 8 emotion classes (fear, contempt) were dropped —
  too few samples (21 and 9 images) to split reliably. This mirrors the
  "Data Imbalances" limitation your report already discusses.
- The RNN treats each image as 48 rows fed sequentially into an LSTM —
  a common lightweight way student projects apply RNNs to *static*
  images (true video/temporal emotion data wasn't part of this dataset).
- Numbers will vary slightly run-to-run unless you fix all random seeds
  identically (already done here: seed=42 throughout).
