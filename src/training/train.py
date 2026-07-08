"""
train.py — Siamese Network Training Orchestrator
================================================

Responsibility:
    Orchestrates the complete training pipeline for the Siamese Network.
    Wires together data loading, model forward passes, loss computation,
    backpropagation, learning rate scheduling, model checkpointing, and
    early stopping into a single, cohesive Trainer class.

    This module owns the training lifecycle. It does NOT implement any
    data, model, or loss logic itself — it delegates entirely to the
    purpose-built modules in src/data/ and src/models/.

Design Pattern:
    Trainer class encapsulates all mutable training state:
        - Current best validation loss (for checkpointing)
        - Early stopping patience counter
        - Full loss history (for plotting and analysis)
        - Device reference and DataLoaders

    This avoids global state and makes the trainer independently testable,
    resumable, and extensible (e.g., adding LR warmup or mixed precision).

Usage:
    from src.training.train import Trainer

    trainer = Trainer()
    history = trainer.train()
    # history = {'train_loss': [...], 'val_loss': [...], 'lr': [...]}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.config.settings import Settings, get_settings
from src.data.dataset_loader import IdentityScanner, SiamesePairDataset
from src.data.pair_generator import PairGenerator
from src.models.contrastive_loss import ContrastiveLoss
from src.models.siamese_network import SiameseNetwork
from src.utils.image_utils import get_transform

logger = logging.getLogger(__name__)

# Type alias for the loss history dictionary
TrainingHistory = Dict[str, List[float]]


# ---------------------------------------------------------------------------
# Device detection helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Detect and return the best available compute device.

    Priority order:
        1. CUDA  — NVIDIA GPU (fastest, fully supported by PyTorch)
        2. MPS   — Apple Silicon GPU (macOS, PyTorch >= 1.12)
        3. CPU   — Universal fallback

    Returns:
        torch.device: The selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU. Training will be slower — consider a GPU.")
    return device


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Orchestrates the full Siamese Network training lifecycle.

    Responsibility:
        Build all necessary components (datasets, loaders, model, optimizer,
        scheduler, loss) from Settings, then execute the training loop with
        early stopping and best-model checkpointing.

    State managed:
        _best_val_loss      — tracks the lowest validation loss seen so far
        _patience_counter   — counts epochs without validation improvement
        _train_losses       — per-epoch training loss history
        _val_losses         — per-epoch validation loss history
        _lr_history         — per-epoch learning rate history

    Usage:
        trainer = Trainer()                     # reads all config from Settings
        history = trainer.train()               # runs full training
        trainer = Trainer(cfg=custom_settings)  # override settings if needed
    """

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        """
        Initialise the Trainer and build all training components.

        This constructor performs all expensive setup (dataset scanning,
        pair generation, model construction) so that train() can immediately
        start the epoch loop without setup overhead.

        Args:
            cfg: Optional Settings override. If None, get_settings() is used.
                 Useful for experimentation without modifying the config file.
        """
        self._cfg    = cfg or get_settings()
        self._device = get_device()

        # ── Reproducibility ─────────────────────────────────────────────────
        torch.manual_seed(self._cfg.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._cfg.RANDOM_SEED)

        # ── Data pipeline ────────────────────────────────────────────────────
        logger.info("Building data pipeline...")
        self._train_loader, self._val_loader = self._build_data_loaders()

        # ── Model, loss, optimiser, scheduler ───────────────────────────────
        logger.info("Building model and training components...")
        self._model     = SiameseNetwork().to(self._device)
        self._criterion = ContrastiveLoss(margin=self._cfg.CONTRASTIVE_MARGIN)
        self._optimizer = optim.Adam(
            self._model.parameters(),
            lr=self._cfg.LEARNING_RATE,
        )
        self._scheduler = StepLR(
            self._optimizer,
            step_size=self._cfg.LR_STEP_SIZE,
            gamma=self._cfg.LR_GAMMA,
        )

        # ── Training state ───────────────────────────────────────────────────
        self._best_val_loss:    float      = float("inf")
        self._patience_counter: int        = 0
        self._train_losses:     List[float] = []
        self._val_losses:       List[float] = []
        self._lr_history:       List[float] = []

        logger.info(
            f"Trainer ready — "
            f"device: {self._device}, "
            f"train batches: {len(self._train_loader)}, "
            f"val batches: {len(self._val_loader)}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> TrainingHistory:
        """
        Execute the full training loop.

        For each epoch:
            1. Run training pass   → compute mean train loss
            2. Run validation pass → compute mean val loss
            3. Step LR scheduler
            4. Log epoch summary (epoch, losses, LR, elapsed time)
            5. Save checkpoint if val loss improved (best-model-only)
            6. Check early stopping condition

        Stops when:
            - NUM_EPOCHS is reached, OR
            - Val loss has not improved for EARLY_STOPPING_PATIENCE epochs

        Returns:
            TrainingHistory: Dict with keys 'train_loss', 'val_loss', 'lr'.
                             Each value is a list with one entry per epoch.
                             Use this for plotting the loss curve.
        """
        logger.info("=" * 62)
        logger.info("  Starting Siamese Network Training")
        logger.info("=" * 62)
        logger.info(f"  Epochs   : {self._cfg.NUM_EPOCHS}")
        logger.info(f"  Patience : {self._cfg.EARLY_STOPPING_PATIENCE}")
        logger.info(f"  Margin   : {self._cfg.CONTRASTIVE_MARGIN}")
        logger.info(f"  LR       : {self._cfg.LEARNING_RATE}")
        logger.info("=" * 62)

        self._print_header()

        for epoch in range(1, self._cfg.NUM_EPOCHS + 1):
            epoch_start = time.perf_counter()

            # ── Training pass ────────────────────────────────────────────
            train_loss = self._run_epoch(epoch, phase="train")

            # ── Validation pass ──────────────────────────────────────────
            val_loss = self._run_epoch(epoch, phase="val")

            # ── Record history ───────────────────────────────────────────
            current_lr = self._optimizer.param_groups[0]["lr"]
            self._train_losses.append(train_loss)
            self._val_losses.append(val_loss)
            self._lr_history.append(current_lr)

            # ── Step LR scheduler (after recording current LR) ───────────
            self._scheduler.step()

            elapsed = time.perf_counter() - epoch_start

            # ── Log epoch summary ────────────────────────────────────────
            improved = val_loss < self._best_val_loss
            self._print_epoch_row(
                epoch, train_loss, val_loss, current_lr, elapsed, improved
            )

            # ── Checkpoint if improved ───────────────────────────────────
            if improved:
                self._best_val_loss    = val_loss
                self._patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self._patience_counter += 1

            # ── Early stopping check ─────────────────────────────────────
            if self._patience_counter >= self._cfg.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"\n⏹  Early stopping triggered after {epoch} epochs. "
                    f"Val loss did not improve for "
                    f"{self._cfg.EARLY_STOPPING_PATIENCE} consecutive epochs."
                )
                break

        self._print_footer()
        logger.info(
            f"Training complete. "
            f"Best val loss: {self._best_val_loss:.6f} "
            f"| Checkpoint: {self._cfg.BEST_MODEL_PATH}"
        )

        return {
            "train_loss": self._train_losses,
            "val_loss":   self._val_losses,
            "lr":         self._lr_history,
        }

    # ------------------------------------------------------------------
    # Epoch runners
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, phase: str) -> float:
        """
        Run one full pass through the training or validation DataLoader.

        Args:
            epoch: Current epoch number (1-indexed), used for tqdm display.
            phase: Either 'train' or 'val'. Controls model mode and whether
                   gradients are computed.

        Returns:
            float: Mean loss over all batches in this epoch.
        """
        is_training = phase == "train"
        self._model.train() if is_training else self._model.eval()

        loader      = self._train_loader if is_training else self._val_loader
        total_loss  = 0.0
        num_batches = len(loader)

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            for img1, img2, labels in loader:
                # Move tensors to compute device
                img1   = img1.to(self._device, non_blocking=True)
                img2   = img2.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)

                if is_training:
                    self._optimizer.zero_grad(set_to_none=True)

                # Forward pass
                emb1, emb2 = self._model(img1, img2)
                loss        = self._criterion(emb1, emb2, labels)

                if is_training:
                    loss.backward()
                    # Gradient clipping: prevents exploding gradients
                    # especially useful early in training when embeddings
                    # are random and losses can be large.
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        max_norm=1.0,
                    )
                    self._optimizer.step()

                total_loss += loss.item()

        return total_loss / num_batches

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save a model checkpoint to disk.

        Only called when the current val_loss is strictly lower than the
        previous best — ensuring the file always contains the best weights.

        The checkpoint includes:
            - model_state_dict   : all learnable parameters
            - optimizer_state_dict: Adam moments (for resuming training)
            - epoch              : epoch at which best was achieved
            - best_val_loss      : the validation loss at this checkpoint
            - train_losses       : full training loss history so far
            - val_losses         : full validation loss history so far

        Args:
            epoch:    The current epoch number.
            val_loss: The current (improved) validation loss.
        """
        checkpoint = {
            "epoch":               epoch,
            "best_val_loss":       val_loss,
            "model_state_dict":    self._model.state_dict(),
            "optimizer_state_dict":self._optimizer.state_dict(),
            "train_losses":        self._train_losses,
            "val_losses":          self._val_losses,
        }

        save_path = self._cfg.BEST_MODEL_PATH
        torch.save(checkpoint, save_path)
        logger.debug(
            f"Checkpoint saved → {save_path} "
            f"(epoch {epoch}, val_loss={val_loss:.6f})"
        )

    def load_checkpoint(self, path: Optional[str | Path] = None) -> int:
        """
        Load a previously saved checkpoint into the model and optimizer.

        Useful for resuming an interrupted training run or for fine-tuning
        from a pre-trained checkpoint.

        Args:
            path: Path to the checkpoint file. If None, uses BEST_MODEL_PATH
                  from settings.

        Returns:
            int: The epoch at which the checkpoint was saved (use as the
                 starting epoch for resuming training).

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        load_path = Path(path) if path else self._cfg.BEST_MODEL_PATH

        if not load_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {load_path}\n"
                "Train a model first before attempting to load."
            )

        checkpoint = torch.load(load_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        epoch = checkpoint.get("epoch", 0)
        logger.info(
            f"Checkpoint loaded from '{load_path}' "
            f"(epoch {epoch}, best_val_loss={self._best_val_loss:.6f})"
        )
        return epoch

    # ------------------------------------------------------------------
    # Data pipeline construction
    # ------------------------------------------------------------------

    def _build_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Scan the dataset, generate pairs, and build train/val DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        # Discover identities
        scanner      = IdentityScanner(self._cfg.DATASET_DIR)
        identity_map = scanner.scan()

        # Generate balanced pairs and split
        generator                = PairGenerator(identity_map)
        train_pairs, val_pairs   = generator.generate()

        # Build Datasets with appropriate augmentation
        train_transform = get_transform(self._cfg.IMAGE_SIZE, augment=True)
        val_transform   = get_transform(self._cfg.IMAGE_SIZE, augment=False)

        train_dataset = SiamesePairDataset(train_pairs, transform=train_transform)
        val_dataset   = SiamesePairDataset(val_pairs,   transform=val_transform)

        # Determine DataLoader worker config based on device
        # pin_memory only benefits GPU transfers; on CPU it adds overhead
        use_pin_memory = self._device.type == "cuda"
        num_workers    = self._cfg.NUM_WORKERS if self._device.type == "cuda" else 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=False,
        )

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        """Print the training progress table header."""
        header = (
            f"\n{'Epoch':>6} │ "
            f"{'Train Loss':>10} │ "
            f"{'Val Loss':>10} │ "
            f"{'LR':>10} │ "
            f"{'Time (s)':>8} │ "
            f"{'Best':>5}"
        )
        separator = "─" * len(header)
        print(separator)
        print(header)
        print(separator)

    def _print_epoch_row(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        elapsed: float,
        improved: bool,
    ) -> None:
        """Print one row of the training progress table."""
        marker = "✅" if improved else "  "
        row = (
            f"{epoch:>6} │ "
            f"{train_loss:>10.6f} │ "
            f"{val_loss:>10.6f} │ "
            f"{lr:>10.2e} │ "
            f"{elapsed:>8.2f} │ "
            f"{marker:>5}"
        )
        print(row)

    def _print_footer(self) -> None:
        """Print the training progress table footer."""
        print("─" * 62)

    # ------------------------------------------------------------------
    # Read-only properties (for evaluate.py and visualization.py)
    # ------------------------------------------------------------------

    @property
    def model(self) -> SiameseNetwork:
        """The trained SiameseNetwork instance."""
        return self._model

    @property
    def device(self) -> torch.device:
        """The compute device in use."""
        return self._device

    @property
    def history(self) -> TrainingHistory:
        """Loss and LR history from the last train() call."""
        return {
            "train_loss": self._train_losses,
            "val_loss":   self._val_losses,
            "lr":         self._lr_history,
        }

    @property
    def best_val_loss(self) -> float:
        """Best validation loss achieved during training."""
        return self._best_val_loss
