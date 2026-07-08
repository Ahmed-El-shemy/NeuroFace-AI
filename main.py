"""
main.py — Siamese Face Recognition System — Entry Point
=========================================================

This is the unified command-line interface for the entire system.
All modes are accessed through this single file.

Modes
-----
  train     Train the Siamese Network from scratch (or resume).
            Saves the best model to models/best_siamese_model.pth.
            Generates training curves to outputs/training_curves.png.

  evaluate  Load the best saved model and run evaluation on the val set.
            Prints metrics table. Generates distance distribution plot.

  build-db  Build the identity embedding database from the dataset using
            the trained model. Saves to outputs/embeddings_database.pkl.

  run       Start the real-time webcam recognition session.
            Requires a trained model and a built embedding database.

  all       Run: train → evaluate → build-db in sequence.
            The complete pipeline in one command.

Usage Examples
--------------
  python main.py --mode train
  python main.py --mode evaluate
  python main.py --mode build-db
  python main.py --mode run
  python main.py --mode all
  python main.py --mode train --epochs 50
  python main.py --mode run --threshold 0.6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — before any project imports so all modules inherit it
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Reduce noise from third-party loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with all flags.
    """
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Siamese Face Recognition System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                # train from scratch
  python main.py --mode train --epochs 100  # override epoch count
  python main.py --mode evaluate            # evaluate saved model
  python main.py --mode build-db            # rebuild embedding database
  python main.py --mode run                 # launch webcam recognition
  python main.py --mode run --threshold 0.6 # with custom threshold
  python main.py --mode all                 # train → evaluate → build-db
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "build-db", "run", "all"],
        required=True,
        help="Operation mode to execute.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override NUM_EPOCHS from settings (train mode only).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override RECOGNITION_THRESHOLD (run/evaluate modes).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plot files (faster for CI/headless).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_train(args: argparse.Namespace) -> None:
    """
    Mode: train
    Train the Siamese Network from scratch using the configured settings.
    Saves the best model checkpoint and optionally generates loss curve plots.
    """
    _banner("TRAINING MODE")

    from src.training.train import Trainer
    from src.utils.visualization import plot_training_curves

    # Optionally patch NUM_EPOCHS via CLI argument
    trainer = Trainer()

    if args.epochs is not None:
        logger.info(f"Overriding NUM_EPOCHS: {trainer._cfg.NUM_EPOCHS} → {args.epochs}")
        # Patch the private counter — Settings is frozen; we use a workaround
        # by stopping the loop early via the trainer's epoch limit check.
        object.__setattr__(trainer._cfg, "NUM_EPOCHS", args.epochs)

    t0      = time.perf_counter()
    history = trainer.train()
    elapsed = time.perf_counter() - t0

    _banner("TRAINING COMPLETE")
    logger.info(f"Total training time : {elapsed / 60:.1f} minutes")
    logger.info(f"Best val loss       : {trainer.best_val_loss:.6f}")
    logger.info(f"Checkpoint          : {trainer._cfg.BEST_MODEL_PATH}")

    if not args.no_plots:
        plot_training_curves(history)
        logger.info("Training curves saved to outputs/training_curves.png")


def run_evaluate(args: argparse.Namespace) -> None:
    """
    Mode: evaluate
    Load the best checkpoint, run evaluation on the validation set,
    print the full metrics table, and generate the distance distribution plot.
    """
    _banner("EVALUATION MODE")

    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset_loader import IdentityScanner, SiamesePairDataset
    from src.data.pair_generator import PairGenerator
    from src.training.evaluate import Evaluator, load_model_for_eval
    from src.utils.image_utils import get_transform
    from src.utils.visualization import plot_distance_distribution
    from src.config.settings import get_settings

    cfg    = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_for_eval(device=device)

    # Rebuild val loader (same pairs, deterministic — same seed → same split)
    scanner      = IdentityScanner(cfg.DATASET_DIR)
    identity_map = scanner.scan()
    _, val_pairs = PairGenerator(identity_map).generate()
    val_ds       = SiamesePairDataset(
        val_pairs, transform=get_transform(cfg.IMAGE_SIZE, augment=False)
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=0)

    threshold = args.threshold or cfg.RECOGNITION_THRESHOLD
    evaluator = Evaluator(model, device=device)
    result    = evaluator.evaluate(val_loader, threshold=threshold if args.threshold else None)

    if not args.no_plots:
        plot_distance_distribution(result)
        logger.info("Distance distribution saved to outputs/distance_distribution.png")


def run_build_db(args: argparse.Namespace) -> None:
    """
    Mode: build-db
    Build the identity embedding database using the trained model and save it.
    Run this every time the dataset or trained model changes.
    """
    _banner("BUILD EMBEDDING DATABASE")

    import torch
    from src.training.evaluate import load_model_for_eval
    from src.recognition.embedding_database import EmbeddingDatabase

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_for_eval(device=device)

    db = EmbeddingDatabase.build(model=model, device=device)
    db.save()

    logger.info(
        f"Database built: {db.size} identities → "
        f"{db.identities}"
    )


def run_webcam(args: argparse.Namespace) -> None:
    """
    Mode: run
    Start the real-time webcam face recognition session.
    Requires: trained model checkpoint + built embedding database.
    Press 'Q' to quit.
    """
    _banner("REAL-TIME RECOGNITION  [Press Q to quit]")

    import torch
    from src.training.evaluate import load_model_for_eval
    from src.recognition.embedding_database import EmbeddingDatabase
    from src.recognition.recognizer import Recognizer
    from src.recognition.webcam import WebcamRecognition, create_detector
    from src.config.settings import get_settings

    cfg    = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and database
    model    = load_model_for_eval(device=device)
    database = EmbeddingDatabase.load()

    threshold = args.threshold or cfg.RECOGNITION_THRESHOLD
    recognizer = Recognizer(model=model, database=database,
                            device=device, threshold=threshold)

    # Build detector (auto-downloads YuNet model if needed)
    detector = create_detector()

    webcam = WebcamRecognition(recognizer=recognizer, detector=detector)
    webcam.run()


def run_all(args: argparse.Namespace) -> None:
    """
    Mode: all
    Full pipeline: train → evaluate → build-db.
    Does NOT launch the webcam — run 'python main.py --mode run' after.
    """
    _banner("FULL PIPELINE: train → evaluate → build-db")
    run_train(args)
    run_evaluate(args)
    run_build_db(args)
    _banner("ALL DONE")
    logger.info("Run 'python main.py --mode run' to start the webcam session.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    """Print a formatted section banner to stdout."""
    line  = "═" * 62
    inner = f"  {text}  "
    pad   = "═" * ((62 - len(inner)) // 2)
    print(f"\n{line}")
    print(f"{pad}{inner}{pad}")
    print(f"{line}\n")


def _check_dataset() -> None:
    """
    Verify the dataset directory exists and has at least 2 identities
    with images. Exits with a clear error message if not.
    """
    from src.config.settings import get_settings
    cfg = get_settings()

    if not cfg.DATASET_DIR.exists():
        logger.error(
            f"Dataset directory not found: {cfg.DATASET_DIR}\n"
            "Create the directory and add one sub-folder per identity,\n"
            "each containing at least 2 face images."
        )
        sys.exit(1)

    identities = [
        d for d in cfg.DATASET_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    if len(identities) < 2:
        logger.error(
            f"Need at least 2 identity folders in {cfg.DATASET_DIR}. "
            f"Found: {len(identities)}."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point — parse args and dispatch to the correct mode."""
    args = parse_args()

    logger.info(f"Mode: {args.mode}")

    # Validate dataset existence for modes that need it
    if args.mode in ("train", "evaluate", "build-db", "all"):
        _check_dataset()

    dispatch = {
        "train":    run_train,
        "evaluate": run_evaluate,
        "build-db": run_build_db,
        "run":      run_webcam,
        "all":      run_all,
    }

    try:
        dispatch[args.mode](args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.exception(f"Unexpected error in mode '{args.mode}': {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
