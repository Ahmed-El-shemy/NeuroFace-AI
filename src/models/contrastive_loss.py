"""
contrastive_loss.py — Contrastive Loss Function
================================================

Responsibility:
    Implements the Contrastive Loss function (Hadsell, Chopra & LeCun, 2006)
    from scratch using PyTorch primitives. This is the training signal that
    teaches the Siamese Network to map same-identity faces close together and
    different-identity faces far apart in the embedding space.

─────────────────────────────────────────────────────────────────────────────
MATHEMATICAL FOUNDATION
─────────────────────────────────────────────────────────────────────────────

Formula:
    L(y, D) = (1 − y) · D²  +  y · max(0, margin − D)²

Where:
    y      ∈ {0, 1}   — pair label
                          0 → POSITIVE pair (same identity)
                          1 → NEGATIVE pair (different identities)
    D                 — Euclidean distance between two L2-normalised embeddings
    margin            — minimum required separation for negative pairs (default: 1.0)

CASE 1 — Positive pair (y = 0, same person):
    L = (1 − 0) · D²  +  0 · (...)
    L = D²

    Interpretation: The network is penalised proportionally to the SQUARE
    of how far apart the embeddings are. Loss → 0 as D → 0. The gradient
    pushes both embeddings toward each other, forcing the network to extract
    the same features regardless of which specific photo it sees.

CASE 2 — Negative pair (y = 1, different people):
    L = (1 − 1) · D²  +  1 · max(0, margin − D)²
    L = max(0, margin − D)²

    Interpretation: The network is penalised ONLY when D < margin.
    Once D ≥ margin, the pair contributes ZERO loss and ZERO gradient.
    This is the key stability mechanism — without the max(0, ...) clamp,
    the loss would be −D², causing the network to push embeddings to
    infinity, leading to gradient explosion and unstable training.

WHY THE MARGIN MATTERS:
    The margin is a hyperparameter that defines the target separation between
    different-identity embeddings. It acts as a "satisfaction threshold":

        D < margin → still too close → apply gradient pressure to separate them
        D ≥ margin → far enough apart → release gradient pressure (loss = 0)

    Because our embeddings are L2-normalised (on the unit hypersphere),
    the maximum possible Euclidean distance is 2.0. A margin of 1.0 places
    the target halfway between "identical" (0.0) and "maximally different" (2.0).
    The recognition threshold (0.5) then sits comfortably in the resulting gap.

NUMERICAL STABILITY:
    The euclidean_distance method adds ε = 1e-8 inside the sqrt to prevent
    NaN gradients when D = 0 (which can occur for identical embeddings in the
    early stages of training, before the network has learned to separate them).

BATCH REDUCTION:
    The mean over the batch is used (not sum) to make the loss magnitude
    independent of batch size. This allows changing BATCH_SIZE in settings
    without needing to re-tune the learning rate.

─────────────────────────────────────────────────────────────────────────────
Usage:
    from src.models.contrastive_loss import ContrastiveLoss

    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(embeddings1, embeddings2, labels)
    loss.backward()
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Network training.

    Responsibility:
        Compute the scalar loss value for a batch of embedding pairs, using
        the Contrastive Loss formula. Acts as the bridge between the network's
        geometric output (distances) and the training signal (gradients).

    Label Convention (must match SiamesePairDataset):
        0 → Positive pair  (same identity)  → loss = D²
        1 → Negative pair  (different ids)  → loss = max(0, margin − D)²

    Usage:
        criterion = ContrastiveLoss()              # uses margin from settings
        criterion = ContrastiveLoss(margin=0.8)    # override margin

        loss = criterion(emb1, emb2, labels)
        loss.backward()
    """

    def __init__(self, margin: float | None = None) -> None:
        """
        Initialise the ContrastiveLoss with a configurable margin.

        Args:
            margin: The separation margin for negative pairs.
                    If None, the value from Settings.CONTRASTIVE_MARGIN is used.
                    Typical range: [0.5, 2.0].
                    Increase if negative pairs are not being separated enough.
                    Decrease if training is unstable (loss oscillating).
        """
        super().__init__()

        cfg = get_settings()
        self.margin: float = margin if margin is not None else cfg.CONTRASTIVE_MARGIN

        logger.info(f"ContrastiveLoss initialised with margin={self.margin:.3f}")

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the mean Contrastive Loss for a batch of embedding pairs.

        Applies the formula:
            L_i = (1 − y_i) · D_i²  +  y_i · max(0, margin − D_i)²
            L    = mean(L_i)  for i in batch

        The mean reduction makes the loss magnitude independent of batch size,
        so BATCH_SIZE can be changed in settings without retuning learning rate.

        Args:
            embeddings1: Embeddings from the first image in each pair.
                         Shape: [B, embedding_dim], dtype: float32.
                         Must be L2-normalized (output of SiameseNetwork).
            embeddings2: Embeddings from the second image in each pair.
                         Shape: [B, embedding_dim], dtype: float32.
                         Must be L2-normalized (output of SiameseNetwork).
            labels:      Pair labels. Shape: [B], dtype: float32.
                         0.0 → positive pair (same identity).
                         1.0 → negative pair (different identities).

        Returns:
            torch.Tensor: Scalar loss value (mean over the batch), dtype float32.
                          Differentiable with respect to embeddings1 and embeddings2.

        Raises:
            ValueError: If tensor shapes are inconsistent.
        """
        if embeddings1.shape != embeddings2.shape:
            raise ValueError(
                f"Embedding shapes must match: "
                f"{embeddings1.shape} vs {embeddings2.shape}"
            )
        if labels.shape[0] != embeddings1.shape[0]:
            raise ValueError(
                f"Batch size mismatch: labels has {labels.shape[0]} elements "
                f"but embeddings have batch size {embeddings1.shape[0]}."
            )

        # ── Step 1: Compute Euclidean distances ─────────────────────────────
        # Shape: [B]   — one distance per pair in the batch
        # ε inside sqrt prevents NaN gradients when distance = 0
        distances = self._euclidean_distance(embeddings1, embeddings2)

        # ── Step 2: Positive pair loss term ─────────────────────────────────
        # Active only when y = 0 (positive pairs)
        # Formula: (1 − y) · D²
        # As D → 0, this term → 0. Gradient pushes embeddings together.
        positive_loss = (1.0 - labels) * distances.pow(2)  # [B]

        # ── Step 3: Negative pair loss term ─────────────────────────────────
        # Active only when y = 1 (negative pairs)
        # Formula: y · max(0, margin − D)²
        # F.relu(margin − D) is equivalent to max(0, margin − D)
        # When D ≥ margin: relu output = 0, loss = 0, gradient = 0 → released
        # When D < margin: penalise to push the pair further apart
        margin_distance = torch.clamp(self.margin - distances, min=0.0)  # [B]
        negative_loss   = labels * margin_distance.pow(2)                # [B]

        # ── Step 4: Combine and reduce ───────────────────────────────────────
        # Per-sample loss: each sample contributes exactly ONE of the two terms
        # (the other is zeroed out by the label multiplier)
        per_sample_loss = positive_loss + negative_loss  # [B]

        # Mean reduction: makes loss scale-invariant to batch size
        loss = per_sample_loss.mean()

        return loss

    @staticmethod
    def _euclidean_distance(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample Euclidean distance between two embedding batches.

        This is a local copy of the distance logic (also available as
        SiameseNetwork.euclidean_distance) to keep ContrastiveLoss self-
        contained and avoid circular imports.

        Formula: D_i = ‖emb1_i − emb2_i‖₂ = sqrt(Σ(emb1 − emb2)² + ε)

        The epsilon (1e-8) is added INSIDE the square root, not to the
        squared distance. This correctly prevents the sqrt gradient from
        becoming undefined at D=0 while keeping the distance mathematically
        accurate (ε adds only ~3e-4 to D when D=0).

        Args:
            emb1: Embedding tensor, shape [B, D].
            emb2: Embedding tensor, shape [B, D].

        Returns:
            torch.Tensor: Per-sample Euclidean distances, shape [B].
        """
        diff    = emb1 - emb2                      # [B, D]
        sq_dist = torch.sum(diff.pow(2), dim=1)    # [B]
        return torch.sqrt(sq_dist + 1e-8)          # [B]

    def extra_repr(self) -> str:
        """String representation for model.print() and logging."""
        return f"margin={self.margin}"
