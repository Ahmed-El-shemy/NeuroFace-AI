"""
settings.py — Central Configuration Module
==========================================

Responsibility:
    Acts as the single source of truth for every constant, path, and
    hyperparameter used across the entire system. No module should ever
    hard-code a value that belongs here.

Design Decision:
    Using a frozen dataclass ensures the configuration is immutable at
    runtime — preventing accidental mutation from any module. All paths
    are resolved using pathlib.Path, making the project fully portable
    across operating systems.

Usage:
    from src.config.settings import get_settings
    cfg = get_settings()
    print(cfg.DATASET_DIR)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Internal helper — resolve project root once at import time
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    """
    Traverse upward from this file to locate the project root.

    The project root is identified by the presence of 'pyproject.toml'
    or 'main.py'. This makes the configuration portable regardless of
    where the interpreter is invoked from.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """
    anchor = Path(__file__).resolve()
    for parent in anchor.parents:
        if (parent / "pyproject.toml").exists() or (parent / "main.py").exists():
            return parent
    raise FileNotFoundError(
        "Could not locate the project root. "
        "Ensure 'pyproject.toml' or 'main.py' exists at the project root."
    )


_PROJECT_ROOT: Path = _find_project_root()


# ---------------------------------------------------------------------------
# Settings Dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """
    Immutable configuration container for the entire Face Recognition system.

    Responsibility:
        Centralizes all paths, model hyperparameters, training parameters,
        and recognition thresholds into one place. Every module reads from
        this object; nothing is hard-coded elsewhere.

    Sections:
        - Project Paths
        - Dataset Configuration
        - Model Architecture
        - Training Hyperparameters
        - Recognition / Inference
        - Face Detection
        - Logging

    Usage:
        cfg = get_settings()
        model_save_path = cfg.BEST_MODEL_PATH
    """

    # ------------------------------------------------------------------
    # Project Paths
    # ------------------------------------------------------------------

    PROJECT_ROOT: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def DATASET_DIR(self) -> Path:
        """Root directory containing one sub-folder per identity."""
        return self.PROJECT_ROOT / "dataset"

    @property
    def MODELS_DIR(self) -> Path:
        """Directory where trained model checkpoints are saved."""
        return self.PROJECT_ROOT / "models"

    @property
    def OUTPUTS_DIR(self) -> Path:
        """Directory for outputs: embeddings database, plots, etc."""
        return self.PROJECT_ROOT / "outputs"

    @property
    def LOGS_DIR(self) -> Path:
        """Directory for log files."""
        return self.PROJECT_ROOT / "logs"

    @property
    def BEST_MODEL_PATH(self) -> Path:
        """Full path to the best saved Siamese Network checkpoint."""
        return self.MODELS_DIR / "best_siamese_model.pth"

    @property
    def EMBEDDINGS_DB_PATH(self) -> Path:
        """Full path to the serialized embeddings database (pickle)."""
        return self.OUTPUTS_DIR / "embeddings_database.pkl"

    @property
    def TRAINING_PLOT_PATH(self) -> Path:
        """Full path to the training/validation loss curve plot."""
        return self.OUTPUTS_DIR / "training_loss_curve.png"

    # ------------------------------------------------------------------
    # Dataset Configuration
    # ------------------------------------------------------------------

    IMAGE_SIZE: tuple[int, int] = (105, 105)
    """
    Target (height, width) to which every face crop is resized.
    105×105 is the canonical size from the original Siamese Network paper
    (Koch et al., 2015) and works well for faces.
    """

    SUPPORTED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    """File extensions recognised as valid images when scanning the dataset."""

    TRAIN_SPLIT: float = 0.8
    """Fraction of generated pairs used for training; rest goes to validation."""

    RANDOM_SEED: int = 42
    """Global random seed for reproducibility across NumPy, PyTorch, and Python."""

    # ------------------------------------------------------------------
    # Model Architecture
    # ------------------------------------------------------------------

    EMBEDDING_DIM: int = 128
    """
    Dimensionality of the embedding vector produced by the Siamese Network.
    128 dimensions provides a rich feature space while remaining efficient
    for distance comparisons.
    """

    CNN_OUT_CHANNELS: tuple[int, ...] = (64, 128, 128, 256)
    """
    Number of output channels for each convolutional block in the shared CNN.
    The network progressively increases depth to capture increasingly abstract
    facial features.
    """

    DROPOUT_RATE: float = 0.3
    """
    Dropout probability applied before the final embedding layer.
    Helps prevent overfitting when the dataset is small.
    """

    # ------------------------------------------------------------------
    # Contrastive Loss
    # ------------------------------------------------------------------

    CONTRASTIVE_MARGIN: float = 1.0
    """
    Margin (m) in the Contrastive Loss formula.

    Contrastive Loss = (1 - y) * D²  +  y * max(0, m - D)²

    Where:
        y = 1 for a NEGATIVE pair (different identities)
        y = 0 for a POSITIVE pair (same identity)
        D = Euclidean distance between embeddings

    The margin forces negative pairs to be at least `m` apart in
    embedding space. A value of 1.0 is the standard starting point.
    """

    # ------------------------------------------------------------------
    # Training Hyperparameters
    # ------------------------------------------------------------------

    BATCH_SIZE: int = 32
    """Number of pairs per training batch."""

    NUM_EPOCHS: int = 50
    """Maximum number of full passes through the training dataset."""

    LEARNING_RATE: float = 1e-3
    """Initial learning rate for the Adam optimizer."""

    LR_STEP_SIZE: int = 10
    """Reduce LR every N epochs (StepLR scheduler)."""

    LR_GAMMA: float = 0.5
    """Multiplicative factor for LR reduction: new_lr = lr * gamma."""

    EARLY_STOPPING_PATIENCE: int = 10
    """
    Stop training if validation loss does not improve for this many epochs.
    Prevents overfitting and wastes compute on a plateaued model.
    """

    NUM_WORKERS: int = 4
    """Number of parallel DataLoader worker processes."""

    PIN_MEMORY: bool = True
    """
    Pin CPU memory for faster GPU data transfer.
    Automatically disabled if CUDA is unavailable.
    """

    # ------------------------------------------------------------------
    # Pair Generation
    # ------------------------------------------------------------------

    PAIRS_PER_IDENTITY: int = 10
    """
    Number of positive AND negative pairs generated per identity.
    Total pairs = num_identities × PAIRS_PER_IDENTITY × 2.
    Increase for larger datasets or more training signal.
    """

    # ------------------------------------------------------------------
    # Recognition / Inference
    # ------------------------------------------------------------------

    RECOGNITION_THRESHOLD: float = 0.5
    """
    Euclidean distance threshold for identity decision.

    If distance(query_embedding, db_embedding) < RECOGNITION_THRESHOLD:
        → Identity RECOGNISED  (green bounding box)
    Else:
        → UNKNOWN              (red bounding box)

    Tuning guide:
        Lower  → stricter matching, more "Unknown" results
        Higher → more permissive, risk of false positives
    """

    # ------------------------------------------------------------------
    # Face Detection
    # ------------------------------------------------------------------

    YUNET_MODEL_PATH: str = ""
    """
    Path to the YuNet ONNX model file.
    Leave empty to trigger automatic Haar Cascade fallback.
    Download from:
    https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    """

    FACE_CONFIDENCE_THRESHOLD: float = 0.6
    """Minimum detection confidence for a face bounding box to be accepted."""

    FACE_NMS_THRESHOLD: float = 0.3
    """Non-Maximum Suppression threshold for overlapping face detections."""

    MIN_FACE_SIZE: int = 30
    """
    Minimum face bounding-box dimension (pixels).
    Faces smaller than this are discarded — they are too small to yield
    reliable embeddings.
    """

    WEBCAM_INDEX: int = 0
    """Index of the camera device to open (0 = default/built-in camera)."""

    WEBCAM_WIDTH: int = 1280
    """Requested webcam frame width in pixels."""

    WEBCAM_HEIGHT: int = 720
    """Requested webcam frame height in pixels."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    LOG_LEVEL: int = logging.INFO
    """Root logger level. Change to logging.DEBUG for verbose output."""

    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    """Unified log format applied to both file and console handlers."""

    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """
    Return the global, shared Settings instance (singleton pattern).

    Using a singleton ensures all modules share the same configuration
    object, making runtime overrides consistent and predictable.

    Returns:
        Settings: The immutable, project-wide configuration object.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _ensure_directories(_settings_instance)
    return _settings_instance


def _ensure_directories(cfg: Settings) -> None:
    """
    Create all required output directories if they do not already exist.

    Called once during settings initialisation. Uses exist_ok=True so
    it is safe to call multiple times without raising errors.

    Args:
        cfg: The fully initialised Settings object.
    """
    dirs_to_create = [
        cfg.MODELS_DIR,
        cfg.OUTPUTS_DIR,
        cfg.LOGS_DIR,
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
