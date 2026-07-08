"""
siamese_network.py — Custom Siamese Neural Network Architecture
==============================================================

Responsibility:
    Defines the complete Siamese Network architecture built entirely from
    scratch using PyTorch primitives. No pretrained models or external
    face recognition libraries are used.

    Three classes with increasing levels of abstraction:

    1. ConvBlock
       A single reusable convolutional unit: Conv2d → BatchNorm2d → ReLU → MaxPool2d.
       Acts as the fundamental building block of the CNN backbone.

    2. EmbeddingNet
       The shared CNN backbone + embedding head. Takes a single face image
       and produces a 128-dimensional L2-normalized embedding vector.
       This is the network that LEARNS what makes a face unique.

    3. SiameseNetwork
       The Siamese wrapper. Holds one EmbeddingNet and applies it twice
       (once per input image) using SHARED WEIGHTS. Returns both embeddings.
       The distance between them is computed externally (in ContrastiveLoss
       and Recognizer) — the network itself does not decide similarity.

Architecture:
    Input [B, 3, 105, 105]
        │
        ▼
    ConvBlock 1: Conv(3→64, 3×3)   + BN + ReLU + MaxPool(2×2)  → [B, 64,  52, 52]
    ConvBlock 2: Conv(64→128, 3×3) + BN + ReLU + MaxPool(2×2)  → [B, 128, 26, 26]
    ConvBlock 3: Conv(128→128,3×3) + BN + ReLU + MaxPool(2×2)  → [B, 128, 13, 13]
    ConvBlock 4: Conv(128→256,3×3) + BN + ReLU + MaxPool(2×2)  → [B, 256,  6,  6]
        │
        ▼
    AdaptiveAvgPool2d(4×4)                                       → [B, 256,  4,  4]
    Flatten                                                       → [B, 4096]
    Dropout(0.3)
    Linear(4096 → 128)                                            → [B, 128]
    L2 Normalize                                                  → [B, 128]  ← unit sphere

Why L2 Normalization:
    Projecting embeddings onto the unit hypersphere bounds the maximum
    Euclidean distance between any two embeddings to exactly 2.0.
    This makes the recognition threshold interpretable:
        distance < 0.5  → same person   (close on the sphere)
        distance ≥ 0.5  → unknown       (far apart on the sphere)
    Without normalization, embeddings can grow unboundedly, making it
    impossible to choose a stable threshold.

Why Shared Weights:
    Both images pass through the SAME EmbeddingNet instance. This forces
    the network to learn a single, consistent representation space. If
    the weights were not shared, the network could cheat by learning two
    different transformations that happen to produce similar vectors for
    the same person but without any meaningful geometric structure.

Usage:
    from src.models.siamese_network import SiameseNetwork

    model = SiameseNetwork()
    emb1, emb2 = model(img_tensor_1, img_tensor_2)
    dist = SiameseNetwork.euclidean_distance(emb1, emb2)
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConvBlock — fundamental building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    A single convolutional processing unit in the CNN backbone.

    Responsibility:
        Encapsulates the standard pattern:
            Conv2d → BatchNorm2d → ReLU → MaxPool2d

        Using BatchNorm after every conv layer:
            - Normalises activations, reducing internal covariate shift
            - Allows higher learning rates without divergence
            - Acts as a mild regularizer (reduces need for Dropout in conv layers)

    Usage:
        block = ConvBlock(in_channels=3, out_channels=64, kernel_size=3)
        output = block(input_tensor)  # spatial dims halved by MaxPool
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        pool_size: int = 2,
    ) -> None:
        """
        Initialise a convolutional block.

        Args:
            in_channels:  Number of input feature channels.
            out_channels: Number of output feature channels (filters).
            kernel_size:  Convolution kernel size. Default 3 (3×3).
            padding:      Zero-padding applied to both sides of the input.
                          Default 1 preserves spatial dimensions before pooling.
            pool_size:    MaxPool2d kernel and stride size. Default 2 halves
                          spatial dimensions: (H, W) → (H/2, W/2).
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,  # bias is redundant when followed by BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conv → BN → ReLU → MaxPool pipeline.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            torch.Tensor: Output tensor of shape [B, out_channels, H/2, W/2].
        """
        return self.block(x)


# ---------------------------------------------------------------------------
# EmbeddingNet — the shared CNN backbone
# ---------------------------------------------------------------------------

class EmbeddingNet(nn.Module):
    """
    Shared CNN backbone that maps a face image to a 128-dim embedding vector.

    Responsibility:
        Learn a mapping f: Image → ℝ¹²⁸ such that faces of the same person
        map to nearby points, and faces of different people map to distant
        points on the unit hypersphere.

        This is the ONLY network with learnable parameters in the Siamese
        system. It is instantiated ONCE and called TWICE per forward pass
        (once per input image) to guarantee weight sharing.

    Architecture:
        4 ConvBlocks with progressively increasing filter counts
        → AdaptiveAvgPool2d(4×4) [spatial size invariant]
        → Flatten
        → Dropout
        → Linear(4096 → EMBEDDING_DIM)
        → L2 Normalize

    Usage:
        net = EmbeddingNet()
        embedding = net(face_tensor)  # [B, 128], L2-normalized
    """

    # The spatial output size fed to AdaptiveAvgPool before the linear layer.
    # (4, 4) gives 256 * 4 * 4 = 4096 features entering the FC layer.
    _POOL_OUTPUT_SIZE: Tuple[int, int] = (4, 4)

    def __init__(self) -> None:
        """Initialise EmbeddingNet using hyperparameters from Settings."""
        super().__init__()

        cfg = get_settings()
        channels = cfg.CNN_OUT_CHANNELS  # (64, 128, 128, 256)
        emb_dim  = cfg.EMBEDDING_DIM     # 128
        dropout  = cfg.DROPOUT_RATE      # 0.3

        # ── Convolutional backbone ──────────────────────────────────────────
        # Build blocks dynamically from the channel tuple in settings.
        # Input channels start at 3 (RGB). Each subsequent block takes the
        # previous block's output channels as its input.
        in_ch = 3
        conv_blocks = []
        for out_ch in channels:
            conv_blocks.append(ConvBlock(in_channels=in_ch, out_channels=out_ch))
            in_ch = out_ch

        self.backbone = nn.Sequential(*conv_blocks)

        # ── Spatial aggregation ─────────────────────────────────────────────
        # AdaptiveAvgPool makes the FC layer size independent of input image
        # resolution. If IMAGE_SIZE changes in settings, this still works.
        self.pool = nn.AdaptiveAvgPool2d(self._POOL_OUTPUT_SIZE)

        # ── Embedding head ──────────────────────────────────────────────────
        fc_in_features = channels[-1] * self._POOL_OUTPUT_SIZE[0] * self._POOL_OUTPUT_SIZE[1]
        # = 256 * 4 * 4 = 4096

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_in_features, emb_dim),
        )

        # Initialise weights for stable early training
        self._init_weights()

        logger.info(
            f"EmbeddingNet created — "
            f"channels: {channels}, "
            f"FC in: {fc_in_features}, "
            f"embedding dim: {emb_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of face images to L2-normalised embedding vectors.

        Pipeline:
            [B, 3, H, W]
            → backbone (4 conv blocks)         → [B, 256, h, w]
            → adaptive avg pool                → [B, 256, 4, 4]
            → flatten + dropout + linear       → [B, 128]
            → L2 normalize                     → [B, 128]  ‖embedding‖₂ = 1

        Args:
            x: Batch of face images, shape [B, 3, H, W], dtype float32.

        Returns:
            torch.Tensor: L2-normalized embeddings, shape [B, EMBEDDING_DIM].
                          Every vector lies on the unit hypersphere.
        """
        features  = self.backbone(x)       # [B, 256, h, w]
        pooled    = self.pool(features)    # [B, 256, 4, 4]
        embedding = self.embedding_head(pooled)  # [B, 128]

        # L2 normalization — project onto unit hypersphere
        # dim=1 operates across the embedding dimension
        normalized = F.normalize(embedding, p=2, dim=1)

        return normalized

    def _init_weights(self) -> None:
        """
        Apply Kaiming (He) initialization to all Conv2d and Linear layers.

        Kaiming initialization is the recommended strategy for layers
        followed by ReLU activations. It sets the initial variance such
        that the signal neither vanishes nor explodes through deep networks,
        enabling faster convergence from the start of training.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias,   0.0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0.0)


# ---------------------------------------------------------------------------
# SiameseNetwork — the full Siamese wrapper
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """
    Full Siamese Network: one shared EmbeddingNet applied twice per forward pass.

    Responsibility:
        Orchestrate the Siamese architecture. Holds a SINGLE EmbeddingNet
        instance and calls it twice — once for each input image — ensuring
        both images are processed by identical (shared) parameters.

        The network outputs two embedding vectors. The Euclidean distance
        between them is computed externally:
            - During training: in ContrastiveLoss
            - During recognition: in Recognizer

        The SiameseNetwork is intentionally decoupled from the loss and
        the distance metric. This keeps each component independently
        testable and replaceable (e.g. swapping to Triplet Loss later).

    Usage:
        model = SiameseNetwork()
        emb1, emb2 = model(img1, img2)
        distance = SiameseNetwork.euclidean_distance(emb1, emb2)
        # distance shape: [B], values in [0.0, 2.0] (L2-normalized embeddings)
    """

    def __init__(self) -> None:
        """Initialise the Siamese Network with one shared EmbeddingNet."""
        super().__init__()
        self.embedding_net = EmbeddingNet()

        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"SiameseNetwork created — "
            f"total params: {total_params:,}, "
            f"trainable: {trainable:,}"
        )

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run both input images through the shared EmbeddingNet.

        Both calls use the EXACT SAME weights (self.embedding_net is one
        object). This is the defining property of a Siamese architecture:
        the representation space is learned symmetrically for both inputs.

        Args:
            img1: First image batch,  shape [B, 3, H, W], dtype float32.
            img2: Second image batch, shape [B, 3, H, W], dtype float32.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                emb1: Embeddings for img1, shape [B, EMBEDDING_DIM].
                emb2: Embeddings for img2, shape [B, EMBEDDING_DIM].
                Both are L2-normalized (unit vectors on the hypersphere).
        """
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        return emb1, emb2

    def get_embedding(self, img: torch.Tensor) -> torch.Tensor:
        """
        Generate an embedding for a single image (inference helper).

        Used by the Recognizer and EmbeddingDatabase to embed individual
        face crops without needing a dummy second image.

        Args:
            img: Image tensor of shape [B, 3, H, W] or [1, 3, H, W].

        Returns:
            torch.Tensor: L2-normalized embedding, shape [B, EMBEDDING_DIM].
        """
        return self.embedding_net(img)

    @staticmethod
    def euclidean_distance(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Euclidean (L2) distance between two embedding batches.

        Because both embeddings are L2-normalized, the resulting distances
        are bounded in [0.0, 2.0]:
            0.0 → identical embeddings (same person, perfect match)
            2.0 → maximally dissimilar (antipodal points on the hypersphere)

        This method is intentionally a @staticmethod so it can be called
        without a model instance — useful in the Recognizer, EmbeddingDatabase,
        and metrics utilities.

        Formula:
            distance_i = ‖emb1_i − emb2_i‖₂

        Args:
            emb1: Embedding tensor, shape [B, D].
            emb2: Embedding tensor, shape [B, D].

        Returns:
            torch.Tensor: Per-sample distances, shape [B].
                          Each value in [0.0, 2.0] for L2-normalized inputs.
        """
        # Squared difference sum → square root = Euclidean distance
        # Adding eps=1e-8 inside the sqrt prevents NaN gradients when dist=0
        diff    = emb1 - emb2                      # [B, D]
        sq_dist = torch.sum(diff ** 2, dim=1)      # [B]
        return torch.sqrt(sq_dist + 1e-8)          # [B]

    def freeze_backbone(self) -> None:
        """
        Freeze all convolutional backbone parameters.

        When called, only the embedding head (Linear layer) will be updated
        during training. Useful for fine-tuning experiments where you want
        to keep early-layer features fixed.

        Note: Not used in the default training pipeline — provided for
        experimentation. Call model.unfreeze_backbone() to reverse.
        """
        for param in self.embedding_net.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — only embedding head will be trained.")

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters for full end-to-end training.

        Args: None
        Returns: None
        """
        for param in self.embedding_net.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full end-to-end training enabled.")
