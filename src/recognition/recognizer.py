"""
recognizer.py — Real-Time Face Recognition Inference Engine
===========================================================

Responsibility:
    Bridges the OpenCV detection pipeline with the Siamese Network and
    EmbeddingDatabase to produce recognition decisions for individual face
    crops in real time.

    Given a raw BGR face crop from OpenCV, the Recognizer:
        1. Preprocesses the crop (BGR → RGB → resize → normalize → tensor)
        2. Generates an L2-normalized embedding via the trained model
        3. Queries the EmbeddingDatabase for the nearest known identity
        4. Returns a typed RecognitionResult with all decision metadata

    Designed to process multiple faces per frame simultaneously via
    recognize_batch(), enabling recognition of an arbitrary number of
    people visible at the same time.

Design Decisions:
    - The Recognizer is intentionally decoupled from EmbeddingDatabase.
      The database is a pure data store; the Recognizer handles all
      image processing and model inference. (Single Responsibility)
    - All inference runs under torch.no_grad() with model.eval() to
      prevent any gradient computation at runtime.
    - Confidence is defined as 1 - D/2.0, mapping the [0, 2] distance
      range to a [0, 1] confidence range. This is purely a display
      convenience — the threshold-based decision is always authoritative.

Usage:
    from src.recognition.recognizer import Recognizer
    from src.recognition.embedding_database import EmbeddingDatabase
    from src.training.evaluate import load_model_for_eval

    model    = load_model_for_eval()
    database = EmbeddingDatabase.load()
    recognizer = Recognizer(model=model, database=database)

    # Single face crop (BGR numpy array from OpenCV):
    result = recognizer.recognize(face_crop_bgr)
    print(result.display_name, result.confidence)

    # Multiple faces per frame:
    results = recognizer.recognize_batch([crop1, crop2, crop3])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from src.config.settings import get_settings
from src.models.siamese_network import SiameseNetwork
from src.recognition.embedding_database import EmbeddingDatabase
from src.utils.image_utils import get_transform, preprocess_face

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RecognitionResult — per-face decision container
# ---------------------------------------------------------------------------

@dataclass
class RecognitionResult:
    """
    Encapsulates the complete recognition decision for one detected face.

    Responsibility:
        Provides a clean, typed interface to all recognition outputs. The
        webcam overlay module (webcam.py) consumes this to decide bounding
        box colour, label text, and confidence display.

    Fields:
        name         : Raw identity name from the database (e.g. 'ahmed_gadalla'),
                       or None if the face is not recognised.
        distance     : Euclidean distance between the query embedding and the
                       nearest identity's mean embedding. Range: [0.0, 2.0].
        threshold    : The decision threshold used for this recognition.
        is_known     : True if distance < threshold (recognised identity).
        confidence   : Display confidence score in [0.0, 1.0].
                       Defined as max(0, 1 - distance / 2.0).
                       This is a display metric only — the authoritative
                       decision is always `is_known` (threshold-based).
        display_name : Human-readable label for the UI overlay.
                       Converts 'ahmed_gadalla' → 'Ahmed Gadalla'.
                       Returns 'Unknown' if is_known is False.
    """

    name:       Optional[str]
    distance:   float
    threshold:  float
    is_known:   bool
    confidence: float
    display_name: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute the display_name from name and is_known."""
        if self.is_known and self.name is not None:
            # 'ahmed_gadalla' → 'Ahmed Gadalla'
            self.display_name = " ".join(
                word.capitalize() for word in self.name.split("_")
            )
        else:
            self.display_name = "Unknown"

    def __str__(self) -> str:
        status = "KNOWN" if self.is_known else "UNKNOWN"
        return (
            f"[{status}] {self.display_name} "
            f"(dist={self.distance:.4f}, "
            f"conf={self.confidence:.2%}, "
            f"threshold={self.threshold:.4f})"
        )


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------

class Recognizer:
    """
    Real-time face recognition inference engine.

    Responsibility:
        Accept raw BGR face crops from the OpenCV detection pipeline,
        preprocess them, generate embeddings, and query the embedding
        database to produce RecognitionResult objects.

    State:
        _model     : SiameseNetwork in eval() mode for embedding generation.
        _database  : EmbeddingDatabase for identity lookup.
        _transform : Inference-mode image transform pipeline.
        _threshold : Recognition distance threshold (from Settings).
        _device    : Compute device (CPU/CUDA/MPS).

    Usage:
        recognizer = Recognizer(model=model, database=database)
        result  = recognizer.recognize(face_crop_bgr)
        results = recognizer.recognize_batch([crop1, crop2])
    """

    def __init__(
        self,
        model:     SiameseNetwork,
        database:  EmbeddingDatabase,
        device:    Optional[torch.device] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Initialise the Recognizer with a trained model and populated database.

        Args:
            model:     Trained SiameseNetwork (loaded from best checkpoint).
                       Will be moved to `device` and set to eval() mode.
            database:  Populated EmbeddingDatabase (loaded from .pkl file).
            device:    Compute device. Defaults to CUDA if available, else CPU.
            threshold: Recognition distance threshold override. If None, uses
                       Settings.RECOGNITION_THRESHOLD.
        """
        cfg = get_settings()

        self._threshold = threshold if threshold is not None else cfg.RECOGNITION_THRESHOLD
        self._device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._model = model
        self._model.to(self._device)
        self._model.eval()

        self._database  = database
        self._transform = get_transform(image_size=cfg.IMAGE_SIZE, augment=False)

        logger.info(
            f"Recognizer ready — "
            f"device: {self._device}, "
            f"threshold: {self._threshold:.4f}, "
            f"identities: {database.size}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recognize(self, face_crop_bgr: np.ndarray) -> RecognitionResult:
        """
        Recognise the identity in a single face crop.

        Full pipeline for one face:
            BGR NumPy crop
            → preprocess (BGR→RGB, resize, normalize)
            → model.get_embedding()
            → database.find_closest()
            → RecognitionResult

        Args:
            face_crop_bgr: Face image in OpenCV BGR format, shape (H, W, 3).
                           Should be a tight crop around the detected face.
                           Any resolution is accepted — resize is applied
                           internally by the transform pipeline.

        Returns:
            RecognitionResult: Complete recognition decision for this face.
        """
        embedding = self._generate_embedding(face_crop_bgr)
        return self._build_result(embedding)

    def recognize_batch(
        self,
        face_crops_bgr: List[np.ndarray],
    ) -> List[RecognitionResult]:
        """
        Recognise identities for a list of face crops simultaneously.

        Processes each crop independently. Designed to handle an arbitrary
        number of faces visible in a single webcam frame. Results are
        returned in the same order as the input crops.

        Implementation note: each crop is processed sequentially because
        the dataset is small (7 identities) and batching the model forward
        pass would add complexity with negligible benefit on CPU. On a GPU
        with many simultaneous faces, batching could be added here.

        Args:
            face_crops_bgr: List of face image arrays in OpenCV BGR format.
                            Each element has shape (H_i, W_i, 3) — crops
                            may have different sizes, as each is resized
                            independently by the transform.

        Returns:
            List[RecognitionResult]: One result per input crop, in order.
                                     Empty list if face_crops_bgr is empty.
        """
        if not face_crops_bgr:
            return []

        results = []
        for crop in face_crops_bgr:
            result = self.recognize(crop)
            results.append(result)

        known_count = sum(1 for r in results if r.is_known)
        logger.debug(
            f"Batch recognition: {len(results)} faces, "
            f"{known_count} recognised, {len(results) - known_count} unknown."
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_embedding(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR face crop and generate its L2-normalised embedding.

        Handles the full OpenCV→PyTorch→numpy conversion pipeline:
            BGR (H,W,3) → PIL RGB → transform → tensor [1,3,H,W]
            → model.get_embedding() → squeeze → numpy [EMBEDDING_DIM]

        Args:
            face_crop_bgr: Face crop in OpenCV BGR format (H, W, 3).

        Returns:
            np.ndarray: L2-normalised embedding vector, shape [EMBEDDING_DIM].

        Raises:
            ValueError: If the crop is not a valid 3-channel array.
        """
        # preprocess_face handles: BGR→RGB, PIL convert, transform, unsqueeze
        tensor = preprocess_face(face_crop_bgr, self._transform)  # [1, 3, H, W]
        tensor = tensor.to(self._device)

        with torch.no_grad():
            embedding = self._model.get_embedding(tensor)          # [1, EMBEDDING_DIM]

        # Detach from computation graph and convert to numpy
        embedding_np = embedding.squeeze(0).cpu().numpy()         # [EMBEDDING_DIM]
        return embedding_np

    def _build_result(self, embedding: np.ndarray) -> RecognitionResult:
        """
        Query the database with an embedding and build a RecognitionResult.

        Args:
            embedding: L2-normalised query embedding, shape [EMBEDDING_DIM].

        Returns:
            RecognitionResult: Complete decision for this embedding.
        """
        name, distance, _ = self._database.find_closest(
            embedding, threshold=self._threshold
        )

        is_known   = name is not None
        # Confidence: linearly maps [0, 2] distance range to [1, 0] confidence
        # Clamped to [0, 1] — distance can theoretically slightly exceed 2.0
        # due to floating-point precision near antipodal points.
        confidence = float(np.clip(1.0 - distance / 2.0, 0.0, 1.0))

        return RecognitionResult(
            name       = name,
            distance   = distance,
            threshold  = self._threshold,
            is_known   = is_known,
            confidence = confidence,
        )

    # ------------------------------------------------------------------
    # Runtime configuration (adjustable without rebuilding)
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current recognition distance threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """
        Update the recognition threshold at runtime.

        Useful for live tuning during webcam sessions — increase to be
        more permissive (fewer Unknowns), decrease to be stricter
        (more Unknowns but fewer false positives).

        Args:
            value: New threshold value. Must be in (0.0, 2.0].
        """
        if not (0.0 < value <= 2.0):
            raise ValueError(
                f"Threshold must be in (0.0, 2.0], got {value}."
            )
        logger.info(f"Recognition threshold updated: {self._threshold:.4f} → {value:.4f}")
        self._threshold = value

    @property
    def database(self) -> EmbeddingDatabase:
        """The embedding database used for identity lookup."""
        return self._database

    @property
    def device(self) -> torch.device:
        """The compute device in use."""
        return self._device
