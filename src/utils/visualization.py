"""
visualization.py — Training and Evaluation Visualisations
==========================================================

Responsibility:
    Generate publication-quality plots from training history and evaluation
    results. All plots are saved as PNG files to the outputs/ directory.

    Functions:
        plot_training_curves(history)         → loss + LR subplots
        plot_distance_distribution(result)    → pos vs neg distance histograms

Design Decisions:
    - All functions return the matplotlib Figure object so callers can
      further customise or embed in a GUI if needed.
    - 'dark_background' style is used for a clean, premium aesthetic that
      matches modern ML tooling (TensorBoard, W&B, etc.).
    - Non-interactive backend (Agg) is forced when a display is unavailable
      (e.g., SSH session), preventing crashes on headless machines.
    - All file I/O is handled internally — callers do not need to know
      about the output directory.

Usage:
    from src.utils.visualization import plot_training_curves, plot_distance_distribution

    history = trainer.history
    fig1    = plot_training_curves(history)

    result  = evaluator.evaluate(val_loader)
    fig2    = plot_distance_distribution(result)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

# Force non-interactive backend before importing pyplot.
# This prevents crashes on headless servers (no display) while still
# allowing plot saving. Must happen before 'import matplotlib.pyplot'.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history:   dict,
    save_path: Optional[str | Path] = None,
    title:     str = "Siamese Network — Training History",
) -> plt.Figure:
    """
    Plot training and validation loss curves, plus the learning rate schedule.

    Layout:
        Top subplot    : Train loss (solid blue) + Val loss (solid orange)
                         with grid, legend, and epoch markers.
        Bottom subplot : Learning rate over epochs (dashed purple).

    Args:
        history:   Dict with keys 'train_loss', 'val_loss', 'lr'.
                   Each value is a list with one float per epoch.
        save_path: Output path for the PNG file. If None, saves to
                   Settings.OUTPUTS_DIR / 'training_curves.png'.
        title:     Figure title.

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    cfg       = get_settings()
    save_path = Path(save_path) if save_path else cfg.OUTPUTS_DIR / "training_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    train_loss = history.get("train_loss", [])
    val_loss   = history.get("val_loss",   [])
    lr         = history.get("lr",         [])
    epochs     = list(range(1, len(train_loss) + 1))

    if not epochs:
        logger.warning("Empty training history — skipping plot.")
        return plt.figure()

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(12, 8), dpi=120)
        fig.suptitle(title, fontsize=15, fontweight="bold",
                     color="white", y=0.98)

        gs = gridspec.GridSpec(2, 1, hspace=0.45, height_ratios=[3, 1])
        ax_loss = fig.add_subplot(gs[0])
        ax_lr   = fig.add_subplot(gs[1])

        # ── Loss subplot ─────────────────────────────────────────────────
        ax_loss.plot(
            epochs, train_loss,
            color="#4C9BE8", linewidth=2.0, marker="o", markersize=4,
            label="Train Loss",
        )
        if val_loss:
            ax_loss.plot(
                epochs, val_loss,
                color="#F0A500", linewidth=2.0, marker="s", markersize=4,
                linestyle="--", label="Val Loss",
            )

        # Best val loss marker
        if val_loss:
            best_epoch = int(np.argmin(val_loss)) + 1
            best_loss  = min(val_loss)
            ax_loss.axvline(
                x=best_epoch, color="#55D688", linewidth=1.2,
                linestyle=":", alpha=0.8,
                label=f"Best Val (ep {best_epoch}: {best_loss:.4f})",
            )
            ax_loss.scatter(
                [best_epoch], [best_loss],
                color="#55D688", s=90, zorder=5,
            )

        ax_loss.set_xlabel("Epoch", fontsize=11, color="#CCCCCC")
        ax_loss.set_ylabel("Contrastive Loss", fontsize=11, color="#CCCCCC")
        ax_loss.set_title("Loss Curves", fontsize=12, color="#EEEEEE", pad=8)
        ax_loss.legend(fontsize=9, loc="upper right")
        ax_loss.grid(True, alpha=0.2, linestyle="--")
        ax_loss.tick_params(colors="#AAAAAA")
        ax_loss.spines[:].set_color("#444444")

        # Annotate final values
        if train_loss:
            ax_loss.annotate(
                f"{train_loss[-1]:.4f}",
                xy=(epochs[-1], train_loss[-1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="#4C9BE8",
            )
        if val_loss:
            ax_loss.annotate(
                f"{val_loss[-1]:.4f}",
                xy=(epochs[-1], val_loss[-1]),
                xytext=(5, -12), textcoords="offset points",
                fontsize=8, color="#F0A500",
            )

        # ── LR subplot ───────────────────────────────────────────────────
        if lr:
            ax_lr.plot(
                epochs, lr,
                color="#BB77DD", linewidth=1.5, linestyle="--",
            )
            ax_lr.fill_between(epochs, lr, alpha=0.15, color="#BB77DD")
            ax_lr.set_xlabel("Epoch", fontsize=10, color="#CCCCCC")
            ax_lr.set_ylabel("LR", fontsize=10, color="#CCCCCC")
            ax_lr.set_title("Learning Rate Schedule", fontsize=11,
                            color="#EEEEEE", pad=6)
            ax_lr.grid(True, alpha=0.2, linestyle="--")
            ax_lr.tick_params(colors="#AAAAAA")
            ax_lr.spines[:].set_color("#444444")
            ax_lr.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.2e")
            )

    fig.savefig(save_path, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    logger.info(f"Training curves saved → '{save_path}'")
    return fig


# ---------------------------------------------------------------------------
# plot_distance_distribution
# ---------------------------------------------------------------------------

def plot_distance_distribution(
    result:    object,
    save_path: Optional[str | Path] = None,
    title:     str = "Embedding Distance Distribution — Positive vs Negative Pairs",
) -> plt.Figure:
    """
    Plot histograms of Euclidean distances for positive and negative pairs.

    Shows how well the model separates same-identity pairs (positive, low
    distance) from different-identity pairs (negative, high distance).
    Overlays vertical lines for the configured threshold and optimal threshold.

    A well-trained model will show:
        - Positive pairs clustering near 0 (narrow, leftward histogram)
        - Negative pairs clustered past the threshold (rightward histogram)
        - Minimal overlap between the two distributions

    Args:
        result:    EvaluationResult from Evaluator.evaluate(). Provides
                   pos_distances, neg_distances, threshold, optimal_threshold.
        save_path: Output path. If None, saves to
                   Settings.OUTPUTS_DIR / 'distance_distribution.png'.
        title:     Figure title.

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    cfg       = get_settings()
    save_path = (
        Path(save_path) if save_path
        else cfg.OUTPUTS_DIR / "distance_distribution.png"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pos = result.pos_distances
    neg = result.neg_distances

    if len(pos) == 0 and len(neg) == 0:
        logger.warning("No distance data — skipping distribution plot.")
        return plt.figure()

    n_bins = min(30, max(10, len(pos) // 2))

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
        fig.suptitle(title, fontsize=14, fontweight="bold",
                     color="white", y=1.01)

        # ── Left: overlapping histograms ─────────────────────────────────
        ax = axes[0]
        if len(pos) > 0:
            ax.hist(
                pos, bins=n_bins, range=(0, 2),
                color="#55D688", alpha=0.65, edgecolor="#33AA66",
                label=f"Positive pairs  (n={len(pos)})",
            )
        if len(neg) > 0:
            ax.hist(
                neg, bins=n_bins, range=(0, 2),
                color="#F05050", alpha=0.65, edgecolor="#CC3333",
                label=f"Negative pairs  (n={len(neg)})",
            )

        # Threshold lines
        ax.axvline(
            result.threshold, color="#FFCC44",
            linewidth=2.0, linestyle="--",
            label=f"Config threshold ({result.threshold:.3f})",
        )
        ax.axvline(
            result.optimal_threshold, color="#44CCFF",
            linewidth=2.0, linestyle=":",
            label=f"Optimal threshold ({result.optimal_threshold:.3f})",
        )

        ax.set_xlabel("Euclidean Distance", fontsize=11, color="#CCCCCC")
        ax.set_ylabel("Count", fontsize=11, color="#CCCCCC")
        ax.set_title("Distance Histogram", fontsize=12, color="#EEEEEE")
        ax.set_xlim(0, 2)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.tick_params(colors="#AAAAAA")
        ax.spines[:].set_color("#444444")

        # ── Right: box + strip plot for easy visual comparison ────────────
        ax2 = axes[1]
        data_for_box = []
        labels_box   = []

        if len(pos) > 0:
            data_for_box.append(pos)
            labels_box.append("Positive")
        if len(neg) > 0:
            data_for_box.append(neg)
            labels_box.append("Negative")

        bp = ax2.boxplot(
            data_for_box,
            tick_labels  = labels_box,
            patch_artist = True,
            widths       = 0.4,
            medianprops  = dict(color="white", linewidth=2),
        )
        colors_bp = ["#55D688", "#F05050"]
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for whisker in bp["whiskers"]:
            whisker.set_color("#888888")
        for cap in bp["caps"]:
            cap.set_color("#888888")
        for flier in bp["fliers"]:
            flier.set(marker="o", color="#AAAAAA", alpha=0.5, markersize=4)

        # Scatter jitter points on top
        for i, (data, color) in enumerate(
            zip(data_for_box, ["#55D688", "#F05050"]), start=1
        ):
            jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(data))
            ax2.scatter(
                np.full(len(data), i) + jitter, data,
                alpha=0.5, color=color, s=15, zorder=3,
            )

        ax2.axhline(
            result.threshold, color="#FFCC44",
            linewidth=1.5, linestyle="--",
            label=f"Threshold ({result.threshold:.3f})",
        )
        ax2.set_ylabel("Euclidean Distance", fontsize=11, color="#CCCCCC")
        ax2.set_title("Box Plot", fontsize=12, color="#EEEEEE")
        ax2.set_ylim(0, 2.1)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, linestyle="--")
        ax2.tick_params(colors="#AAAAAA")
        ax2.spines[:].set_color("#444444")

        # ── Shared annotation: stats summary ─────────────────────────────
        stats_text = (
            f"Pos: μ={pos.mean():.3f} σ={pos.std():.3f}\n"
            f"Neg: μ={neg.mean():.3f} σ={neg.std():.3f}\n"
            f"Gap: {neg.mean() - pos.mean():.3f}\n"
            f"Acc: {result.accuracy:.2%}   F1: {result.f1_score:.4f}"
        )
        fig.text(
            0.5, -0.02, stats_text,
            ha="center", fontsize=10, color="#BBBBBB",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1E1E2E",
                      edgecolor="#555555", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    logger.info(f"Distance distribution plot saved → '{save_path}'")
    return fig
