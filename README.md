# Handwritten Character Recognition (MNIST CNN) — CodeAlpha ML Internship (Task 3)

## Problem Statement & Objectives

Classify 28×28 grayscale handwritten digits (`0`–`9`) using a CPU-friendly PyTorch Convolutional Neural Network (CNN) with reproducible training/validation/test workflows, overfitting controls, per-class evaluation, and an interactive Streamlit drawing/upload UI.

## Dataset

- **Name**: MNIST (Modified National Institute of Standards and Technology)
- **Source**: Yann LeCun et al. via `torchvision.datasets.MNIST`
- **Size**: 60,000 training images + 10,000 test images (28×28 grayscale, 10 classes)
- **Splits**: 54,000 train / 6,000 validation (`seed=42`) / 10,000 held-out test

## Technology Stack

- Python 3.12 (`>=3.10`)
- `torch`, `torchvision` (CPU)
- `numpy`, `scikit-learn`, `matplotlib`, `pillow`
- `streamlit`, `streamlit-drawable-canvas`
- `pytest`, `ruff`

## Repository Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
├── app.py                     # Interactive Streamlit drawing & upload UI
├── configs/config.json        # Dataset, training, and evaluation configuration
├── data/                      # MNIST download directory (gitignored)
├── models/                    # Saved DigitCNN weights (gitignored)
├── reports/
│   ├── figures/               # Sample grids, training curves, confusion matrix
│   ├── metrics.json           # Machine-readable evaluation metrics
│   ├── model_evaluation.md    # Detailed evaluation report
│   └── training_history.json  # Per-epoch train/val loss & accuracy
├── scripts/
│   ├── download_data.py       # Downloads MNIST via torchvision
│   └── visualize_samples.py   # Saves representative digit grid & sample PNG
├── src/handwritten_recognition/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```bash
# 1. Download MNIST dataset
python scripts/download_data.py

# 2. Export representative sample grid & sample digit image
python scripts/visualize_samples.py

# 3. Train DigitCNN (5 epochs, early stopping, CPU-friendly)
python -m handwritten_recognition.train

# 4. Evaluate on 10,000 MNIST test images & generate figures
python -m handwritten_recognition.evaluate

# 5. Predict a single digit image
python -m handwritten_recognition.predict --image data/sample_digit.png --no-auto-invert

# 6. Launch optional Streamlit drawing/upload app
streamlit run app.py
```

## CNN Architecture (`DigitCNN`)

- **Block 1**: `Conv2d(1 → 32, 3×3, padding=1)` → `BatchNorm2d(32)` → `ReLU` → `MaxPool2d(2)`
- **Block 2**: `Conv2d(32 → 64, 3×3, padding=1)` → `BatchNorm2d(64)` → `ReLU` → `MaxPool2d(2)` → `Dropout2d(0.25)`
- **Classifier**: `Flatten` → `Linear(3136 → 128)` → `ReLU` → `Dropout(0.4)` → `Linear(128 → 10)`

## Actual Evaluation Results (Held-Out MNIST Test Set, `n=10,000`)

- **Test Accuracy**: `0.9883` (98.83%)
- **Macro Precision**: `0.9882`
- **Macro Recall**: `0.9882`
- **Macro F1-Score**: `0.9882`
- **Weighted F1-Score**: `0.9883`

See [reports/model_evaluation.md](reports/model_evaluation.md) and `reports/figures/` for training curves, confusion matrix, and sample predictions.

## Testing & Code Quality

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

## Limitations & Future Improvements

- Trained on centered 28×28 digits (`0`–`9`) only; arbitrary camera photos may require cropping/deskewing.
- Extendable to letters via EMNIST or full word/sentence recognition via CRNN + CTC loss.

## Author

**Aryan Saxena** — CodeAlpha Machine Learning Internship
