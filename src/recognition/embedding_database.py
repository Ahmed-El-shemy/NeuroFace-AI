"""
embedding_database.py — Identity Embedding Database
====================================================

Responsibility:
    Provides two capabilities:

    1. BUILDING — Scan the dataset, run every image through the trained
       Siamese Network, compute a mean embedding vector per identity, and
       serialise the entire database to disk.

    2. QUERYING — Load the persisted database and find the closest known
       identity to any query embedding vector using Euclidean distance.

Why Mean Embedding:
    Each identity typically has multiple images. Running a live face crop
    against every stored image individually would be O(N_images) and brittle
    — a single blurry photo could corrupt the match. By averaging all
    per-image embeddings into one centroid, we get a noise-reduced
    representative vector that is more stable across lighting and pose
    variation. This reduces lookup complexity to O(N_identities).

Storage Format:
    The database is serialised as a Python pickle (.pkl) file containing a
    dict: {identity_name: EmbeddingRecord}. Pickle is chosen over JSON
    because numpy arrays serialise natively without base64 encoding.

Usage:
    # Build (run once after training):
    from src.recognition.embedding_database import EmbeddingDatabase
    db = EmbeddingDatabase.build(model=model, device=device)
    db.save()

    # Load at runtime:
    db = EmbeddingDatabase.load()
    name, distance, path = db.find_closest(query_embedding)
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config.settings import get_settings
from src.data.dataset_loader import IdentityScanner
from src.models.siamese_network import SiameseNetwork
from src.utils.image_utils import get_transform, load_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingRecord — per-identity data container
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingRecord:
    """
    Stores all embedding data for a single identity.

    Responsibility:
        Acts as the leaf node in the embedding database. Each instance
        corresponds to one person and holds their mean embedding (used
        for fast lookup) and all individual per-image embeddings (used
        for debugging and visualisation).

    Fields:
        name           : Identity name matching the dataset folder name.
        mean_embedding : L2-normalised mean of all per-image embeddings.
                         Shape: [EMBEDDING_DIM]. Used for Euclidean lookup.
        embeddings     : List of per-image embedding arrays, each [EMBEDDING_DIM].
                         Retained for analysis and debugging.
        image_paths    : Corresponding image file paths for each embedding.
                         Same length as `embeddings`.
    """

    name:           str
    mean_embedding: np.ndarray
    embeddings:     List[np.ndarray] = field(repr=False)
    image_paths:    List[Path]       = field(repr=False)

    def __post_init__(self) -> None:
        """Validate that embeddings and paths have matching lengths."""
        if len(self.embeddings) != len(self.image_paths):
            raise ValueError(
                f"EmbeddingRecord '{self.name}': "
                f"embeddings ({len(self.embeddings)}) and "
                f"image_paths ({len(self.image_paths)}) lengths must match."
            )

    @property
    def num_images(self) -> int:
        """Number of images used to build this record."""
        return len(self.embeddings)


# ---------------------------------------------------------------------------
# EmbeddingDatabase
# ---------------------------------------------------------------------------

class EmbeddingDatabase:
    """
    Manages the collection of identity embedding records.

    Responsibility:
        Orchestrates building the embedding database from the dataset and
        a trained model, persisting it to disk, loading it at runtime, and
        answering nearest-neighbour queries for recognition.

    State:
        _records: Dict mapping identity name → EmbeddingRecord.
                  Populated either by build() or load().

    Usage:
        # One-time build after training:
        db = EmbeddingDatabase.build(model, device)
        db.save()

        # At recognition time:
        db = EmbeddingDatabase.load()
        name, dist, path = db.find_closest(query_embedding, threshold=0.5)
    """

    def __init__(self) -> None:
        """Initialise an empty database. Use build() or load() to populate."""
        self._cfg     = get_settings()
        self._records: Dict[str, EmbeddingRecord] = {}

    # ------------------------------------------------------------------
    # Class-level factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model:  SiameseNetwork,
        device: Optional[torch.device] = None,
    ) -> "EmbeddingDatabase":
        """
        Build the embedding database from scratch using the trained model.

        Pipeline for each identity:
            1. Load every valid image from the identity's directory.
            2. Preprocess each image through the inference transform pipeline.
            3. Run through the model's get_embedding() method.
            4. Collect all embeddings into a list.
            5. Compute the L2-normalised mean embedding (the identity centroid).
            6. Store as an EmbeddingRecord.

        The model is set to eval() mode automatically. No gradients are
        computed during embedding generation.

        Args:
            model:  Trained SiameseNetwork instance (loaded from checkpoint).
            device: Compute device. Defaults to CPU if not specified.

        Returns:
            EmbeddingDatabase: A populated database ready for querying or saving.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
        """
        cfg    = get_settings()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        db = cls()

        # Transform for inference: no augmentation
        transform = get_transform(image_size=cfg.IMAGE_SIZE, augment=False)

        # Discover all identities
        scanner      = IdentityScanner(cfg.DATASET_DIR)
        identity_map = scanner.scan()

        logger.info(
            f"Building embedding database for {len(identity_map)} "
            f"identities on device '{device}'..."
        )

        with torch.no_grad():
            for name, image_paths in identity_map.items():
                record = cls._build_record(
                    name, image_paths, model, transform, device
                )
                if record is not None:
                    db._records[name] = record
                    logger.info(
                        f"  [{name}] — {record.num_images} image(s) embedded. "
                        f"Mean embedding norm: "
                        f"{np.linalg.norm(record.mean_embedding):.4f}"
                    )

        logger.info(
            f"Database built: {len(db._records)} identities, "
            f"embedding dim: {cfg.EMBEDDING_DIM}"
        )
        return db

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "EmbeddingDatabase":
        """
        Load a previously saved embedding database from disk.

        Args:
            path: Path to the .pkl file. If None, uses
                  Settings.EMBEDDINGS_DB_PATH automatically.

        Returns:
            EmbeddingDatabase: A fully populated database ready for querying.

        Raises:
            FileNotFoundError: If the database file does not exist.
        """
        cfg      = get_settings()
        load_path = Path(path) if path else cfg.EMBEDDINGS_DB_PATH

        if not load_path.exists():
            raise FileNotFoundError(
                f"Embedding database not found: {load_path}\n"
                "Build the database first: python main.py --mode build-db"
            )

        with open(load_path, "rb") as f:
            records: Dict[str, EmbeddingRecord] = pickle.load(f)

        db          = cls()
        db._records = records

        logger.info(
            f"Embedding database loaded from '{load_path}' "
            f"({len(records)} identities: {list(records.keys())})"
        )
        return db

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str | Path] = None) -> Path:
        """
        Serialise the database to a pickle file.

        Args:
            path: Save path. If None, uses Settings.EMBEDDINGS_DB_PATH.

        Returns:
            Path: The path where the database was saved.

        Raises:
            RuntimeError: If the database is empty (not built or loaded).
        """
        if not self._records:
            raise RuntimeError(
                "Cannot save an empty database. "
                "Call build() or load() first."
            )

        save_path = Path(path) if path else self._cfg.EMBEDDINGS_DB_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self._records, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(
            f"Embedding database saved → '{save_path}' "
            f"({len(self._records)} identities)"
        )
        return save_path

    # ------------------------------------------------------------------
    # Recognition query
    # ------------------------------------------------------------------

    def find_closest(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float, Optional[Path]]:
        """
        Find the identity whose mean embedding is closest to the query.

        Computes Euclidean distance between the query embedding and the
        mean embedding of every identity in the database. Returns the
        identity with the smallest distance if it falls below the threshold.

        Distance bounds (L2-normalised embeddings):
            0.0 → identical embeddings (perfect match)
            2.0 → maximally dissimilar (antipodal on unit hypersphere)

        Args:
            query_embedding: The embedding of the face to recognise.
                             Shape: [EMBEDDING_DIM] or [1, EMBEDDING_DIM].
                             Accepts both numpy arrays and PyTorch tensors.
            threshold:       Euclidean distance cutoff. Distances below this
                             value are recognised; above it → "Unknown".
                             If None, uses Settings.RECOGNITION_THRESHOLD.

        Returns:
            Tuple of:
                name     (str | None) : Identity name if recognised, else None.
                distance (float)      : Euclidean distance to the closest match.
                                        Always returned even for Unknown results.
                best_path(Path | None): Path to the closest matching image.
                                        None for Unknown results.

        Raises:
            RuntimeError: If the database is empty.
        """
        if not self._records:
            raise RuntimeError(
                "The embedding database is empty. "
                "Call build() or load() before querying."
            )

        threshold = threshold if threshold is not None else self._cfg.RECOGNITION_THRESHOLD

        # Normalise input to numpy float32 1-D array
        query = self._to_numpy(query_embedding)

        best_name:     Optional[str]  = None
        best_distance: float          = float("inf")
        best_path:     Optional[Path] = None

        for name, record in self._records.items():
            distance = float(np.linalg.norm(query - record.mean_embedding))
            if distance < best_distance:
                best_distance = distance
                best_name     = name
                # Return the path of the individual image whose embedding is
                # closest to the query (more informative than mean path)
                best_path = self._find_closest_image_path(query, record)

        if best_distance >= threshold:
            # Not close enough to any known identity
            return None, best_distance, None

        return best_name, best_distance, best_path

    def find_top_k(
        self,
        query_embedding: np.ndarray | torch.Tensor,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k closest identities sorted by ascending distance.

        Useful for debugging and visualisation — shows the runner-up
        candidates alongside the winning identity.

        Args:
            query_embedding: Query embedding, shape [EMBEDDING_DIM].
            k:               Number of closest identities to return.

        Returns:
            List[Tuple[str, float]]: List of (name, distance) tuples,
                                     sorted from closest to furthest.
        """
        if not self._records:
            raise RuntimeError("Empty database — call build() or load() first.")

        query   = self._to_numpy(query_embedding)
        results = []

        for name, record in self._records.items():
            dist = float(np.linalg.norm(query - record.mean_embedding))
            results.append((name, dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_record(
        name:        str,
        image_paths: list,
        model:       SiameseNetwork,
        transform:   object,
        device:      torch.device,
    ) -> Optional[EmbeddingRecord]:
        """
        Generate an EmbeddingRecord for a single identity.

        Loads each image, runs it through the model, collects all embeddings,
        then computes and L2-normalises the mean embedding (centroid).

        If all images fail to load (e.g. all corrupted), returns None and
        logs a warning — the identity is skipped rather than crashing.

        Args:
            name:        Identity name (folder name in dataset).
            image_paths: List of valid image Paths for this identity.
            model:       SiameseNetwork in eval() mode.
            transform:   Inference-mode transform pipeline.
            device:      Compute device.

        Returns:
            EmbeddingRecord | None: Record for this identity, or None on failure.
        """
        cfg = get_settings()
        per_image_embeddings: List[np.ndarray] = []
        valid_paths:          List[Path]       = []

        for img_path in image_paths:
            try:
                pil_image = load_image(img_path)
                tensor    = transform(pil_image)               # [3, H, W]
                tensor    = tensor.unsqueeze(0).to(device)     # [1, 3, H, W]

                embedding = model.get_embedding(tensor)        # [1, EMBEDDING_DIM]
                embedding_np = embedding.squeeze(0).cpu().numpy()  # [EMBEDDING_DIM]

                per_image_embeddings.append(embedding_np)
                valid_paths.append(img_path)

            except Exception as exc:
                logger.warning(
                    f"Failed to embed image '{img_path}' for identity '{name}': {exc}"
                )
                continue

        if not per_image_embeddings:
            logger.warning(
                f"No valid embeddings generated for identity '{name}'. Skipping."
            )
            return None

        # Compute mean centroid and L2-normalise it
        # L2-normalisation ensures the mean embedding remains on (or near)
        # the unit hypersphere, keeping distances comparable to the threshold.
        mean_embedding = np.mean(per_image_embeddings, axis=0)  # [EMBEDDING_DIM]
        norm           = np.linalg.norm(mean_embedding)
        if norm > 1e-8:
            mean_embedding = mean_embedding / norm

        return EmbeddingRecord(
            name           = name,
            mean_embedding = mean_embedding.astype(np.float32),
            embeddings     = [e.astype(np.float32) for e in per_image_embeddings],
            image_paths    = valid_paths,
        )

    @staticmethod
    def _find_closest_image_path(
        query:  np.ndarray,
        record: EmbeddingRecord,
    ) -> Path:
        """
        Find the individual image within a record closest to the query.

        When the database records multiple images per identity, this returns
        the path to the best-matching individual image (not just the identity's
        mean). Useful for debugging and displaying in the recognition overlay.

        Args:
            query:  Query embedding, shape [EMBEDDING_DIM].
            record: EmbeddingRecord for the matched identity.

        Returns:
            Path: Path to the closest individual image.
        """
        distances = [
            np.linalg.norm(query - emb)
            for emb in record.embeddings
        ]
        best_idx = int(np.argmin(distances))
        return record.image_paths[best_idx]

    @staticmethod
    def _to_numpy(embedding: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Convert a query embedding to a flat float32 numpy array.

        Accepts both numpy arrays and PyTorch tensors (with or without
        batch dimension), normalising them to shape [EMBEDDING_DIM].

        Args:
            embedding: Embedding in any supported format.

        Returns:
            np.ndarray: Flat float32 array of shape [EMBEDDING_DIM].
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        embedding = np.asarray(embedding, dtype=np.float32)

        if embedding.ndim == 2:          # [1, D] → [D]
            embedding = embedding.squeeze(0)

        return embedding

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def identities(self) -> List[str]:
        """Sorted list of all identity names in the database."""
        return sorted(self._records.keys())

    @property
    def size(self) -> int:
        """Number of identities in the database."""
        return len(self._records)

    @property
    def records(self) -> Dict[str, EmbeddingRecord]:
        """Read-only view of the full records dictionary."""
        return self._records

    def __repr__(self) -> str:
        return (
            f"EmbeddingDatabase("
            f"identities={self.size}, "
            f"names={self.identities})"
        )
