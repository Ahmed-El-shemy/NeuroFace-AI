"""
dataset_loader.py — Identity Scanner and Siamese Pair Dataset
=============================================================

Responsibility:
    Provides two focused components for the data pipeline:

    1. IdentityScanner
       Scans the dataset directory and builds a validated mapping of
       identity names to their corresponding image file paths. This is
       the single point of truth for what data is available.

    2. SiamesePairDataset
       A PyTorch Dataset that serves pre-generated pairs (img1, img2, label)
       as normalised tensors. It does NOT generate pairs itself — pair
       generation is the responsibility of pair_generator.py.
       This strict separation ensures each class has one reason to change.

Design Decisions:
    - Lazy loading: images are loaded from disk in __getitem__, not __init__.
      This keeps memory usage constant regardless of dataset size.
    - The Dataset accepts an external transform so that the caller
      (train.py) controls whether augmentation is applied — the Dataset
      itself has no opinion on this.
    - Corrupted or unreadable images are skipped with a warning rather
      than crashing the entire training run.

Usage:
    from src.data.dataset_loader import IdentityScanner, SiamesePairDataset
    from src.utils.image_utils import get_transform

    scanner = IdentityScanner(dataset_dir)
    identity_map = scanner.scan()   # {'ahmed': [Path, ...], 'ali': [...]}

    transform = get_transform(image_size=(105, 105), augment=True)
    dataset = SiamesePairDataset(pairs=pairs_list, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.config.settings import get_settings
from src.utils.image_utils import get_transform, load_image, validate_image_path

logger = logging.getLogger(__name__)

# Type alias for clarity throughout this module
IdentityMap = Dict[str, List[Path]]
Pair = Tuple[Path, Path, int]  # (img_path_1, img_path_2, label)


# ---------------------------------------------------------------------------
# IdentityScanner
# ---------------------------------------------------------------------------

class IdentityScanner:
    """
    Scans the dataset directory and produces a validated identity map.

    Responsibility:
        Discover all identities (sub-directories) in the dataset root,
        collect all valid image paths for each identity, and report
        any skipped files. Performs no image loading — only path validation.

    The identity map it produces is the shared input to both:
        - pair_generator.py  → for generating training pairs
        - embedding_database.py → for building the recognition database

    Usage:
        scanner = IdentityScanner("/path/to/dataset")
        identity_map = scanner.scan()
        # {'ahmed_gadalla': [PosixPath('...jpg'), ...], 'ehab': [...]}
    """

    def __init__(self, dataset_dir: Optional[str | Path] = None) -> None:
        """
        Initialise the scanner with the dataset root directory.

        Args:
            dataset_dir: Path to the dataset root. If None, the path from
                         Settings is used automatically.
        """
        cfg = get_settings()
        self._dataset_dir = Path(dataset_dir) if dataset_dir else cfg.DATASET_DIR
        self._cfg = cfg

    def scan(self) -> IdentityMap:
        """
        Traverse the dataset directory and build the identity → images map.

        Each immediate sub-directory of dataset_dir is treated as one identity.
        Hidden directories (starting with '.') are ignored. Within each identity
        directory, only files with supported image extensions are collected.
        Files that fail validation are logged as warnings and skipped.

        Returns:
            IdentityMap: Dict mapping identity name (str) to a sorted list of
                         valid image Paths. Identities with zero valid images
                         are excluded and logged as warnings.

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
            ValueError: If fewer than 2 valid identities are found (a Siamese
                        Network requires at least 2 identities to form negative pairs).
        """
        if not self._dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self._dataset_dir}\n"
                "Create the 'dataset/' folder and add one sub-directory per identity."
            )

        logger.info(f"Scanning dataset directory: {self._dataset_dir}")
        identity_map: IdentityMap = {}

        identity_dirs = sorted([
            d for d in self._dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not identity_dirs:
            raise ValueError(
                f"No identity sub-directories found in: {self._dataset_dir}"
            )

        for identity_dir in identity_dirs:
            name = identity_dir.name
            valid_images = self._collect_images(identity_dir, name)

            if not valid_images:
                logger.warning(
                    f"Identity '{name}' has no valid images — skipping."
                )
                continue

            identity_map[name] = valid_images
            logger.info(f"  [{name}] → {len(valid_images)} image(s)")

        if len(identity_map) < 2:
            raise ValueError(
                f"Found only {len(identity_map)} valid identity/identities. "
                "A Siamese Network requires at least 2 identities to generate "
                "negative pairs. Please add more identity directories."
            )

        total_images = sum(len(v) for v in identity_map.values())
        logger.info(
            f"Scan complete: {len(identity_map)} identities, "
            f"{total_images} total images."
        )
        return identity_map

    def _collect_images(self, identity_dir: Path, identity_name: str) -> List[Path]:
        """
        Collect and validate all image files within one identity directory.

        Recursion is intentionally avoided — only files directly inside the
        identity directory are considered. This prevents accidental ingestion
        of nested files (e.g., __MACOSX artefacts from zip extraction).

        Args:
            identity_dir:   Path to the identity's sub-directory.
            identity_name:  Name of the identity (used only for logging).

        Returns:
            List[Path]: Sorted list of valid image paths.
        """
        valid: List[Path] = []

        for file_path in sorted(identity_dir.iterdir()):
            if file_path.is_dir():
                continue  # skip nested directories silently

            if not validate_image_path(file_path):
                if not file_path.name.startswith("."):
                    logger.debug(
                        f"Skipping non-image file: {file_path.name} "
                        f"(identity: {identity_name})"
                    )
                continue

            valid.append(file_path)

        return valid

    @property
    def dataset_dir(self) -> Path:
        """The dataset root directory this scanner is configured for."""
        return self._dataset_dir


# ---------------------------------------------------------------------------
# SiamesePairDataset
# ---------------------------------------------------------------------------

class SiamesePairDataset(Dataset):
    """
    PyTorch Dataset that serves pre-generated (image1, image2, label) pairs.

    Responsibility:
        Load image pairs from disk on demand and return them as normalised
        PyTorch tensors. This class is intentionally dumb — it does not
        decide which pairs to create. That decision belongs to pair_generator.py.

    Label Convention:
        0 → Positive pair  (same identity)
        1 → Negative pair  (different identities)

    This convention follows the original Siamese Network paper (Koch et al.
    2015) and aligns with the Contrastive Loss formula used in this project:
        L = (1 - y) * D²  +  y * max(0, margin - D)²

    Usage:
        pairs = [(path_a, path_b, 0), (path_c, path_d, 1), ...]
        transform = get_transform(image_size=(105, 105), augment=True)
        dataset = SiamesePairDataset(pairs=pairs, transform=transform)
        img1, img2, label = dataset[0]
        # img1.shape → [3, 105, 105]
    """

    def __init__(
        self,
        pairs: List[Pair],
        transform: Optional[object] = None,
    ) -> None:
        """
        Initialise the dataset with a list of pre-computed pairs.

        Args:
            pairs:     List of (path1, path2, label) tuples. Each path is a
                       Path object or a string pointing to a valid image file.
                       label is 0 (positive) or 1 (negative).
            transform: A torchvision transform pipeline (from get_transform()).
                       If None, a default inference-mode transform is applied
                       using the image size from Settings.
        """
        if not pairs:
            raise ValueError("Cannot create a SiamesePairDataset with an empty pairs list.")

        self._pairs = pairs
        self._transform = transform or get_transform(
            image_size=get_settings().IMAGE_SIZE,
            augment=False,
        )

        logger.info(
            f"SiamesePairDataset initialised with {len(pairs)} pairs "
            f"({self._count_label(0)} positive, {self._count_label(1)} negative)."
        )

    def __len__(self) -> int:
        """
        Return the total number of pairs in this dataset.

        Returns:
            int: Number of (img1, img2, label) pairs.
        """
        return len(self._pairs)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and return one pair of preprocessed face tensors.

        Images are loaded lazily (here, not in __init__) to keep memory
        usage constant regardless of dataset size. Corrupted images trigger
        a warning and return the pair with a blank (zero) image to avoid
        crashing the DataLoader.

        Args:
            index: Integer index into the pairs list.

        Returns:
            Tuple of:
                img1  (torch.Tensor): Shape [3, H, W], dtype float32.
                img2  (torch.Tensor): Shape [3, H, W], dtype float32.
                label (torch.Tensor): Scalar tensor, 0.0 or 1.0, dtype float32.
                                      Float is required by ContrastiveLoss.
        """
        path1, path2, label = self._pairs[index]

        img1 = self._safe_load(path1)
        img2 = self._safe_load(path2)

        label_tensor = torch.tensor(float(label), dtype=torch.float32)

        return img1, img2, label_tensor

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_load(self, path: Path) -> torch.Tensor:
        """
        Load an image from disk with error recovery.

        If the image cannot be loaded (corrupted, moved, permission error),
        a warning is logged and a zero tensor is returned in its place.
        This prevents a single bad file from halting an entire training epoch.

        Args:
            path: Path to the image file to load.

        Returns:
            torch.Tensor: Preprocessed image tensor of shape [3, H, W],
                          or a zero tensor of the same shape on failure.
        """
        try:
            pil_image = load_image(path)
            return self._transform(pil_image)
        except Exception as exc:
            cfg = get_settings()
            h, w = cfg.IMAGE_SIZE
            logger.warning(
                f"Failed to load image '{path}': {exc}. "
                f"Returning zero tensor [3, {h}, {w}]."
            )
            return torch.zeros(3, *cfg.IMAGE_SIZE, dtype=torch.float32)

    def _count_label(self, label: int) -> int:
        """Count pairs with a specific label for logging purposes."""
        return sum(1 for _, _, lbl in self._pairs if lbl == label)

    # ------------------------------------------------------------------
    # Public properties (read-only access to internal state)
    # ------------------------------------------------------------------

    @property
    def pairs(self) -> List[Pair]:
        """The full list of (path1, path2, label) pairs."""
        return self._pairs

    @property
    def num_positive(self) -> int:
        """Number of positive pairs (same identity) in this dataset."""
        return self._count_label(0)

    @property
    def num_negative(self) -> int:
        """Number of negative pairs (different identities) in this dataset."""
        return self._count_label(1)
