# Siamese Face Recognition System

A complete, production-quality Real-Time Face Recognition system built from scratch using PyTorch and OpenCV. 

This project demonstrates how to build a deep learning pipeline without relying on pre-built recognition libraries (like `face_recognition` or `dlib`), implementing the core mathematical concepts (Contrastive Loss, L2 Normalization, Euclidean Distance Thresholding) and architectural principles (Clean Architecture, SOLID) from the ground up.

## Features

- **Custom Siamese Neural Network**: CNN backbone generating 128-dimensional L2-normalized embeddings.
- **Contrastive Loss**: Custom implementation with geometric margin and numerical stability protections.
- **Automated Data Pipeline**: Dynamically generates balanced positive/negative image pairs (with sampling replacement) directly from raw folders.
- **Real-Time Webcam Inference**: Uses OpenCV 5's deep-learning `YuNet` detector (auto-downloads if missing) for highly accurate face bounding boxes, combined with the custom Siamese network for identity recognition.
- **Robust Identity Database**: Uses mean-centroid embeddings for each identity to maximize robustness to noise and lighting variations.
- **Threshold Optimization**: Sweeps candidate thresholds during evaluation to find the exact distance threshold that maximizes the F1 Score on the validation set.

---

## Project Architecture

The system is strictly divided into focused modules following the Single Responsibility Principle:

```text
src/
├── config/
│   └── settings.py          # Centralised frozen configuration (hyperparameters, paths)
├── data/
│   ├── dataset_loader.py    # Identity scanning and PyTorch Dataset creation
│   └── pair_generator.py    # Stratified positive/negative pair generation logic
├── models/
│   ├── siamese_network.py   # CNN backbone and distance calculations
│   └── contrastive_loss.py  # L_i = (1-y)D^2 + y*max(0, m-D)^2 implementation
├── training/
│   ├── train.py             # Training loop, LR scheduling, checkpointing
│   └── evaluate.py          # Metric calculation and F1 threshold sweeping
├── recognition/
│   ├── embedding_database.py# Mean-centroid persistence layer
│   ├── recognizer.py        # OpenCV → PyTorch inference bridge
│   └── webcam.py            # Live video loop and YuNet face detection
└── utils/
    ├── image_utils.py       # Augmentations and preprocessing
    └── visualization.py     # Matplotlib training curves and distributions
```

---

## Setup & Installation

**1. Clone and Install Dependencies:**
The system requires Python 3.10+ and the following core dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib
```

**2. Prepare your Dataset:**
Create a folder named `dataset/` in the root directory. Inside it, create one folder per person containing their face images. You need at least 2 identities, with at least 2 images each.

```text
dataset/
├── ahmed/
│   ├── img1.jpg
│   └── img2.jpg
└── ehab/
    ├── img1.jpg
    └── img2.jpg
```

---

## Usage (CLI)

The entire system is controlled through the unified `main.py` entry point.

### 1. Train the Model
Train the Siamese Network from scratch. The best checkpoint is automatically saved to `models/best_siamese_model.pth`.
```bash
python main.py --mode train
```
*To override the default 50 epochs:* `python main.py --mode train --epochs 100`

### 2. Evaluate Performance
Run the validation set to see detailed metrics (Accuracy, Precision, Recall, F1) and generate distance distribution plots.
```bash
python main.py --mode evaluate
```

### 3. Build the Database
Compute the mean embedding for every person in your dataset and save it to the persistent database. **You must run this after training or after adding new photos.**
```bash
python main.py --mode build-db
```

### 4. Run Live Webcam
Launch the real-time recognition interface. (Press `Q` to quit).
```bash
python main.py --mode run
```
*To use a stricter recognition threshold:* `python main.py --mode run --threshold 0.4`

### Full Pipeline Automation
You can run the full preparation pipeline (Train → Evaluate → Build DB) in a single command:
```bash
.venv/bin/python main.py --mode all
.venv/bin/python main.py --mode train --epochs 5

```

---

## Outputs

All generated files are saved automatically:
- **`models/best_siamese_model.pth`**: The trained network weights.
- **`outputs/embeddings_database.pkl`**: The compiled identity database.
- **`outputs/training_curves.png`**: Visualisation of Loss and Learning Rate over time.
- **`outputs/distance_distribution.png`**: Histogram showing how well the model separates known identities from unknown faces.
