"""
image_utils.py — Image Loading and Preprocessing Utilities
===========================================================

Responsibility:
    Provides all image I/O and transformation logic required by the
    Face Recognition pipeline. Acts as a pure utility layer — contains
    zero business logic, only data transformation functions.

    All transformations are deterministic at inference time and
    stochastic at training time (data augmentation).

Design Decisions:
    - PIL is used for loading (not OpenCV) because torchvision.transforms
      operates natively on PIL Images, avoiding redundant conversions.
    - OpenCV is reserved exclusively for face detection and webcam I/O.
    - Normalization uses ImageNet statistics as a strong prior — these
      values are well-studied and generalise well even for faces.
    - Augmentation is kept subtle: aggressive transforms (large crops,
      heavy colour distortion) can destroy discriminative facial features.

Usage:
    from src.utils.image_utils import load_image, get_transform, preprocess_face

    transform = get_transform(image_size=(105, 105), augment=True)
    img = load_image("/path/to/face.jpg")
    tensor = transform(img)  # Shape: [3, 105, 105]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation statistics.
# Using these as a prior is standard practice even for non-ImageNet datasets
# because the statistics are stable and the values generalise well.
# ---------------------------------------------------------------------------
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> Image.Image:
    """
    Load an image from disk and convert it to RGB colour space.

    Always converts to RGB, regardless of the source format (RGBA, L, P, etc.).
    This guarantees a consistent 3-channel tensor downstream.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        PIL.Image.Image: RGB image object.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        image = Image.open(path).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot decode image file: {path}") from exc

    return image


def get_transform(
    image_size: Tuple[int, int] = (105, 105),
    augment: bool = False,
) -> transforms.Compose:
    """
    Build and return a torchvision preprocessing pipeline.

    Two modes are supported:
        augment=False (inference / validation):
            Resize → ToTensor → Normalize
            Fully deterministic. Always produces the same output for
            the same input image.

        augment=True (training):
            RandomHorizontalFlip → ColorJitter → RandomRotation →
            Resize → ToTensor → Normalize
            Stochastic. Introduces controlled variation to reduce
            overfitting on small face datasets.

    Augmentation Strategy:
        - Horizontal flip: Faces are roughly symmetric; this doubles
          effective data without distorting identity features.
        - Color jitter (subtle): Handles lighting variation across
          images captured in different conditions.
        - Small rotation (±10°): Handles head tilt without destroying
          facial geometry.
        - NO aggressive crops or large affine transforms — these can
          discard discriminative facial landmarks (eyes, nose bridge).

    Args:
        image_size: Target (height, width) for resizing. Defaults to
                    (105, 105) per the original Siamese Network paper.
        augment:    If True, adds stochastic augmentation steps suitable
                    for training. If False, deterministic inference mode.

    Returns:
        torchvision.transforms.Compose: Ready-to-use transform pipeline.
    """
    h, w = image_size

    base_transforms = [
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]

    if augment:
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
            ),
            transforms.RandomRotation(degrees=10),
        ]
        # Augmentation goes BEFORE resize so spatial ops work on originals
        pipeline = augmentation_transforms + base_transforms
    else:
        pipeline = base_transforms

    return transforms.Compose(pipeline)


def preprocess_face(
    face_crop_bgr: np.ndarray,
    transform: transforms.Compose,
) -> torch.Tensor:
    """
    Convert an OpenCV BGR face crop to a normalised PyTorch tensor.

    This is the bridge between the OpenCV detection pipeline and the
    PyTorch inference pipeline. Used at recognition time when frames
    arrive from the webcam.

    Pipeline:
        NumPy BGR (H, W, 3)  →  PIL RGB  →  transform  →  Tensor [1, 3, H, W]

    The output includes a batch dimension (unsqueeze(0)) so it can be
    passed directly to the model without additional reshaping.

    Args:
        face_crop_bgr: NumPy uint8 array in OpenCV BGR format, shape (H, W, 3).
        transform:     A preprocessing pipeline returned by get_transform().

    Returns:
        torch.Tensor: Normalised face tensor with shape [1, 3, H, W].

    Raises:
        ValueError: If face_crop_bgr is not a 3-channel NumPy array.
    """
    if face_crop_bgr.ndim != 3 or face_crop_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel (H, W, 3) BGR image, "
            f"got shape {face_crop_bgr.shape}"
        )

    # OpenCV uses BGR; PIL uses RGB — swap channels
    rgb_array = face_crop_bgr[:, :, ::-1].copy()  # HWC BGR → HWC RGB
    pil_image = Image.fromarray(rgb_array.astype(np.uint8))

    tensor = transform(pil_image)      # [3, H, W]
    return tensor.unsqueeze(0)         # [1, 3, H, W]


def tensor_to_numpy(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Convert a normalised PyTorch tensor back to a displayable NumPy image.

    Primarily used for visualisation and debugging — e.g. plotting training
    pairs to confirm the data pipeline is working correctly.

    Args:
        tensor:      Image tensor of shape [3, H, W] or [1, 3, H, W].
                     If a batch dimension is present it is squeezed out.
        denormalize: If True, reverses the ImageNet normalisation so pixel
                     values return to the [0, 1] range before scaling to
                     uint8. Set to False only if the tensor is already
                     in [0, 1] range.

    Returns:
        np.ndarray: uint8 array of shape (H, W, 3) in RGB channel order,
                    suitable for display with matplotlib or PIL.
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)  # [1, 3, H, W] → [3, H, W]

    img = tensor.detach().cpu().float()

    if denormalize:
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        img  = img * std + mean

    img = img.clamp(0.0, 1.0)
    img = (img * 255).byte()
    img = img.permute(1, 2, 0).numpy()  # [3, H, W] → [H, W, 3]

    return img  # HWC RGB uint8


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Swap colour channels from BGR (OpenCV default) to RGB.

    A thin convenience wrapper around the standard channel-reversal idiom.
    Used when passing OpenCV frames to matplotlib or PIL for display.

    Args:
        image_bgr: NumPy array in BGR channel order, shape (H, W, 3).

    Returns:
        np.ndarray: Same array with channels reversed to RGB order.
    """
    return image_bgr[:, :, ::-1].copy()


def validate_image_path(path: str | Path) -> bool:
    """
    Check whether a file path points to a supported image file.

    Used by the dataset loader to filter directory contents and skip
    hidden files, system files, or unsupported formats without raising
    exceptions.

    Args:
        path: Path to validate.

    Returns:
        bool: True if the file exists and has a supported image extension.
    """
    from src.config.settings import get_settings  # local import avoids circular deps
    cfg = get_settings()

    path = Path(path)
    return path.is_file() and path.suffix.lower() in cfg.SUPPORTED_EXTENSIONS
