"""
evaluate.py — Post-Training Evaluation for the Siamese Network
==============================================================

Responsibility:
    After training completes, this module evaluates the quality of the
    learned embedding space by computing:

    1. Per-pair Euclidean distances on the validation set.
    2. Classification metrics (accuracy, precision, recall, F1) at the
       configured recognition threshold.
    3. The OPTIMAL threshold — found by sweeping all candidate values
       and selecting the one that maximises the F1 score.
    4. Distance distribution statistics for positive and negative pairs
       separately, used to assess how well the model separates identities.

Design Decisions:
    - All metrics are implemented manually (no sklearn dependency for metrics)
      to make the mathematics explicit and educational.
    - Results are returned as a typed EvaluationResult dataclass, not as
      a loose dictionary. This gives callers IDE auto-complete, type checking,
      and clear documentation of every field.
    - This module is strictly READ-ONLY — it loads the model, runs inference,
      and returns results. It does not modify model weights or save files.
      Visualization (plots) is handled separately in utils/visualization.py.

Classification convention:
    distance < threshold  → predict SAME PERSON  (label 0)
    distance ≥ threshold  → predict DIFFERENT    (label 1)

Usage:
    from src.training.evaluate import Evaluator, load_model_for_eval

    model = load_model_for_eval()           # loads best checkpoint
    evaluator = Evaluator(model, device)
    result = evaluator.evaluate(val_loader)

    print(f'Accuracy  : {result.accuracy:.4f}')
    print(f'F1 Score  : {result.f1_score:.4f}')
    print(f'Optimal θ : {result.optimal_threshold:.4f}')
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.settings import get_settings
from src.models.siamese_network import SiameseNetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EvaluationResult — typed output container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Container for all post-training evaluation metrics and data.

    Responsibility:
        Bundles every output of the evaluation pipeline into one typed,
        inspectable object. Consumed by visualization.py for plotting
        and by main.py for final reporting.

    Fields:
        distances       : Euclidean distance for every val pair. Shape: [N].
        labels          : Ground-truth labels (0=positive, 1=negative). Shape: [N].
        threshold       : The threshold from settings used for primary metrics.
        accuracy        : Fraction of pairs classified correctly at `threshold`.
        precision       : TP / (TP + FP) — of predicted positives, how many are real.
        recall          : TP / (TP + FN) — of all real positives, how many found.
        f1_score        : Harmonic mean of precision and recall.
        optimal_threshold: Threshold that maximises F1 on the validation set.
        optimal_f1      : F1 score achieved at optimal_threshold.
        pos_distances   : Distances for positive pairs only (same identity).
        neg_distances   : Distances for negative pairs only (different identities).
    """

    # Raw data
    distances: np.ndarray
    labels:    np.ndarray

    # Metrics at the configured threshold
    threshold: float
    accuracy:  float
    precision: float
    recall:    float
    f1_score:  float

    # Optimal threshold search results
    optimal_threshold: float
    optimal_f1:        float

    # Distribution analysis
    pos_distances: np.ndarray = field(repr=False)
    neg_distances: np.ndarray = field(repr=False)

    def __str__(self) -> str:
        """Human-readable summary of evaluation results."""
        lines = [
            "",
            "┌──────────────────────────────────────────────────────┐",
            "│              Evaluation Results Summary               │",
            "├──────────────────────────────┬───────────────────────┤",
            f"│  Threshold (config)          │ {self.threshold:<21.4f} │",
            f"│  Accuracy                    │ {self.accuracy:<21.4f} │",
            f"│  Precision                   │ {self.precision:<21.4f} │",
            f"│  Recall                      │ {self.recall:<21.4f} │",
            f"│  F1 Score                    │ {self.f1_score:<21.4f} │",
            "├──────────────────────────────┼───────────────────────┤",
            f"│  Optimal Threshold (val F1)  │ {self.optimal_threshold:<21.4f} │",
            f"│  Optimal F1                  │ {self.optimal_f1:<21.4f} │",
            "├──────────────────────────────┼───────────────────────┤",
            f"│  Pos. dist mean ± std        │ {self.pos_distances.mean():.4f} ± {self.pos_distances.std():.4f}       │",
            f"│  Neg. dist mean ± std        │ {self.neg_distances.mean():.4f} ± {self.neg_distances.std():.4f}       │",
            f"│  Separation gap              │ {self.neg_distances.mean() - self.pos_distances.mean():<21.4f} │",
            f"│  Total val pairs             │ {len(self.labels):<21} │",
            "└──────────────────────────────┴───────────────────────┘",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates a trained Siamese Network on a validation DataLoader.

    Responsibility:
        Run inference (no gradients) on all validation pairs, compute
        Euclidean distances, then calculate classification metrics at
        the configured threshold and find the optimal threshold.

    Usage:
        model = load_model_for_eval()
        evaluator = Evaluator(model, device=torch.device('cpu'))
        result = evaluator.evaluate(val_loader)
        print(result)
    """

    # Number of threshold candidates in the sweep (0.0 → 2.0)
    _THRESHOLD_STEPS: int = 200

    def __init__(
        self,
        model: SiameseNetwork,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialise the Evaluator with a trained model.

        Args:
            model:  A SiameseNetwork instance (trained or loaded from checkpoint).
                    The model is set to eval() mode automatically.
            device: Compute device. Defaults to CPU if not specified.
        """
        self._cfg    = get_settings()
        self._model  = model
        self._device = device or torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()

    def evaluate(
        self,
        val_loader: DataLoader,
        threshold:  Optional[float] = None,
    ) -> EvaluationResult:
        """
        Run the complete evaluation pipeline on the validation DataLoader.

        Steps:
            1. Forward pass (no gradients) on all batches → collect distances
            2. Compute metrics at the given threshold (or Settings default)
            3. Sweep thresholds → find optimal F1
            4. Split distances by label for distribution analysis
            5. Package and return EvaluationResult

        Args:
            val_loader: DataLoader yielding (img1, img2, label) batches.
                        Should use the INFERENCE transform (augment=False).
            threshold:  Decision boundary override. If None, uses
                        Settings.RECOGNITION_THRESHOLD.

        Returns:
            EvaluationResult: All metrics and raw data for this evaluation run.
        """
        logger.info("Running evaluation on validation set...")

        distances_np, labels_np = self._collect_distances(val_loader)

        # ── Metrics at the configured (or overridden) threshold ──────────────
        threshold = threshold if threshold is not None else self._cfg.RECOGNITION_THRESHOLD
        acc, prec, rec, f1 = self._compute_metrics(
            distances_np, labels_np, threshold
        )

        # ── Optimal threshold search ─────────────────────────────────────────
        opt_threshold, opt_f1 = self._find_optimal_threshold(
            distances_np, labels_np
        )

        # ── Distance distribution by label ───────────────────────────────────
        pos_mask = labels_np == 0
        neg_mask = labels_np == 1
        pos_distances = distances_np[pos_mask]
        neg_distances = distances_np[neg_mask]

        result = EvaluationResult(
            distances          = distances_np,
            labels             = labels_np,
            threshold          = threshold,
            accuracy           = acc,
            precision          = prec,
            recall             = rec,
            f1_score           = f1,
            optimal_threshold  = opt_threshold,
            optimal_f1         = opt_f1,
            pos_distances      = pos_distances,
            neg_distances      = neg_distances,
        )

        logger.info(str(result))
        return result

    # ------------------------------------------------------------------
    # Inference pass
    # ------------------------------------------------------------------

    def _collect_distances(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the model on all batches and collect per-pair distances and labels.

        Args:
            loader: DataLoader yielding (img1, img2, label) tuples.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                distances: Float array of shape [N], Euclidean distances.
                labels:    Int array of shape [N], 0=positive, 1=negative.
        """
        all_distances: List[np.ndarray] = []
        all_labels:    List[np.ndarray] = []

        with torch.no_grad():
            for img1, img2, labels in loader:
                img1   = img1.to(self._device, non_blocking=True)
                img2   = img2.to(self._device, non_blocking=True)

                emb1, emb2 = self._model(img1, img2)
                distances  = SiameseNetwork.euclidean_distance(emb1, emb2)

                all_distances.append(distances.cpu().numpy())
                all_labels.append(labels.numpy().astype(int))

        distances_np = np.concatenate(all_distances)
        labels_np    = np.concatenate(all_labels)

        logger.debug(
            f"Collected {len(distances_np)} pair distances. "
            f"Range: [{distances_np.min():.4f}, {distances_np.max():.4f}]"
        )
        return distances_np, labels_np

    # ------------------------------------------------------------------
    # Metric computation (manual implementation)
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        distances: np.ndarray,
        labels:    np.ndarray,
        threshold: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute accuracy, precision, recall, and F1 at a given threshold.

        Classification rule:
            distance < threshold  → predict 0  (same person / positive)
            distance ≥ threshold  → predict 1  (different / negative)

        Confusion matrix (from the perspective of "positive = same person"):
            TP: predicted 0 AND true label 0  (correctly recognised same person)
            FP: predicted 0 AND true label 1  (incorrectly said different=same)
            FN: predicted 1 AND true label 0  (missed a same-person pair)
            TN: predicted 1 AND true label 1  (correctly said different=different)

        Formulas:
            Accuracy  = (TP + TN) / N
            Precision = TP / (TP + FP)   — how reliable are our "match" decisions?
            Recall    = TP / (TP + FN)   — of all real matches, how many did we catch?
            F1        = 2 × (P × R) / (P + R)  — harmonic mean, penalises imbalance

        Args:
            distances: Per-pair Euclidean distances, shape [N].
            labels:    Ground-truth labels (0 or 1), shape [N].
            threshold: Decision boundary for same/different classification.

        Returns:
            Tuple[float, float, float, float]: (accuracy, precision, recall, f1)
        """
        # distance >= threshold → predict 1 (different/negative)
        # distance <  threshold → predict 0 (same/positive)
        # Using >= produces the prediction label directly (0 or 1),
        # consistent with the label convention: 0=positive, 1=negative.
        predictions = (distances >= threshold).astype(int)

        TP = int(np.sum((predictions == 0) & (labels == 0)))
        FP = int(np.sum((predictions == 0) & (labels == 1)))
        FN = int(np.sum((predictions == 1) & (labels == 0)))
        TN = int(np.sum((predictions == 1) & (labels == 1)))

        N        = len(labels)
        accuracy = (TP + TN) / N if N > 0 else 0.0

        # Guard against division by zero when a class is never predicted
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return accuracy, precision, recall, f1

    def _find_optimal_threshold(
        self,
        distances: np.ndarray,
        labels:    np.ndarray,
    ) -> Tuple[float, float]:
        """
        Find the threshold that maximises F1 score on the validation set.

        Sweeps _THRESHOLD_STEPS candidate thresholds uniformly from 0.0
        to 2.0 (the full valid range for L2-normalized embeddings).
        For each candidate, compute F1 and track the best.

        This is a principled alternative to guessing a threshold — the
        optimal value reflects what the model has actually learned and
        how well the embedding space is structured on THIS dataset.

        IMPORTANT: This threshold is computed on the VALIDATION set.
        In a production system, you would use a separate held-out test
        set to avoid threshold overfitting. For this dataset size, the
        distinction is not critical.

        Args:
            distances: Per-pair Euclidean distances, shape [N].
            labels:    Ground-truth labels (0 or 1), shape [N].

        Returns:
            Tuple[float, float]: (optimal_threshold, best_f1_score).
        """
        candidates = np.linspace(0.0, 2.0, self._THRESHOLD_STEPS)

        best_threshold = self._cfg.RECOGNITION_THRESHOLD
        best_f1        = 0.0

        for t in candidates:
            _, _, _, f1 = self._compute_metrics(distances, labels, t)
            if f1 > best_f1:
                best_f1        = f1
                best_threshold = float(t)

        logger.info(
            f"Optimal threshold: {best_threshold:.4f} "
            f"(F1={best_f1:.4f}) "
            f"vs configured threshold: {self._cfg.RECOGNITION_THRESHOLD:.4f}"
        )
        return best_threshold, best_f1

    def _compute_confusion_matrix(
        self,
        distances: np.ndarray,
        labels: np.ndarray,
        threshold: float,
    ) -> Tuple[int, int, int, int]:
        """
        Compute the confusion matrix components at a given threshold.

        Args:
            distances: Per-pair Euclidean distances.
            labels:    Ground-truth labels (0=positive, 1=negative).
            threshold: Decision boundary.

        Returns:
            Tuple[int, int, int, int]: (TP, FP, FN, TN)
        """
        preds = (distances >= threshold).astype(int)
        TP = int(np.sum((preds == 0) & (labels == 0)))
        FP = int(np.sum((preds == 0) & (labels == 1)))
        FN = int(np.sum((preds == 1) & (labels == 0)))
        TN = int(np.sum((preds == 1) & (labels == 1)))
        return TP, FP, FN, TN


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_model_for_eval(
    checkpoint_path: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> SiameseNetwork:
    """
    Load a trained SiameseNetwork from the best checkpoint file.

    This is a standalone convenience function for use in main.py and
    Jupyter notebooks. It handles the full load + weight restore cycle.

    Args:
        checkpoint_path: Path to the .pth checkpoint. If None, uses
                         Settings.BEST_MODEL_PATH automatically.
        device:          Target device. Defaults to CPU.

    Returns:
        SiameseNetwork: Model with loaded weights, in eval() mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    cfg  = get_settings()
    path = Path(checkpoint_path) if checkpoint_path else cfg.BEST_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at: {path}\n"
            "Run training first: python main.py --mode train"
        )

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model = SiameseNetwork().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch    = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("best_val_loss", float("nan"))

    logger.info(
        f"Model loaded from '{path}' "
        f"(epoch={epoch}, best_val_loss={val_loss:.6f})"
    )
    return model
