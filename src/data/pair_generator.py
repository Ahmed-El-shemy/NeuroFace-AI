"""
pair_generator.py — Balanced Pair Generation for Siamese Training
=================================================================

Responsibility:
    Transforms a raw identity map ({name: [image_paths]}) into two balanced
    lists of (path1, path2, label) pairs: one for training, one for validation.
    This is the sole module responsible for all pair-sampling logic.

    It produces ZERO I/O — no images are loaded here. It only manipulates
    file paths and integer labels, making it fast, testable, and memory-safe.

Label Convention (consistent with dataset_loader.py and contrastive_loss.py):
    0 → Positive pair  (same identity)
    1 → Negative pair  (different identities)

Algorithm:
    Positive pairs:
        For each identity, enumerate all unique (img_i, img_j) combinations.
        If PAIRS_PER_IDENTITY exceeds the unique count (e.g., identities
        with only 2 images yield 1 unique pair), sample with replacement.
        This avoids crashing on small identities while still providing the
        requested training signal.

    Negative pairs:
        For each identity, randomly select PAIRS_PER_IDENTITY anchor images
        from that identity, pairing each with a randomly chosen image from a
        randomly chosen *different* identity. This ensures every identity
        appears as the anchor in an equal number of negative pairs.

    Balance:
        |positive_pairs| == |negative_pairs| guaranteed.
        This is critical for Contrastive Loss — a class imbalance would bias
        the gradient towards the majority pair type.

    Train/Val Split:
        A single global shuffle (seeded for reproducibility) is applied to
        the combined pairs list. The first TRAIN_SPLIT fraction becomes the
        training set; the remainder becomes validation.
        The 50/50 label ratio is preserved in both splits because the shuffle
        acts on a pre-balanced list.

Usage:
    from src.data.pair_generator import PairGenerator
    from src.data.dataset_loader import IdentityScanner

    scanner = IdentityScanner()
    identity_map = scanner.scan()

    generator = PairGenerator(identity_map)
    train_pairs, val_pairs = generator.generate()
"""

from __future__ import annotations

import itertools
import logging
import random
from pathlib import Path
from typing import List, Tuple

from src.config.settings import get_settings
from src.data.dataset_loader import IdentityMap, Pair

logger = logging.getLogger(__name__)


class PairGenerator:
    """
    Generates balanced positive and negative image pairs for Siamese training.

    Responsibility:
        Given a validated identity map, produce reproducible, class-balanced
        (train_pairs, val_pairs) splits. All sampling parameters are read
        from Settings — no hard-coded values.

    Usage:
        generator = PairGenerator(identity_map)
        train_pairs, val_pairs = generator.generate()

        # Inspect the result
        print(generator.summary())
    """

    def __init__(self, identity_map: IdentityMap) -> None:
        """
        Initialise the generator with a pre-validated identity map.

        Args:
            identity_map: Dict from identity name (str) to sorted list of
                          valid image Paths, as returned by IdentityScanner.scan().

        Raises:
            ValueError: If fewer than 2 identities are provided (negative
                        pairs require at least 2 distinct identities).
        """
        if len(identity_map) < 2:
            raise ValueError(
                f"PairGenerator requires at least 2 identities, "
                f"got {len(identity_map)}."
            )

        self._identity_map = identity_map
        self._cfg = get_settings()
        self._rng = random.Random(self._cfg.RANDOM_SEED)

        # Store generated results for post-hoc inspection
        self._train_pairs: List[Pair] = []
        self._val_pairs: List[Pair] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[List[Pair], List[Pair]]:
        """
        Run the full pair generation and train/val split pipeline.

        Steps:
            1. Generate positive pairs for every identity.
            2. Generate an equal number of negative pairs.
            3. Combine and shuffle (seeded).
            4. Split into train / val sets.
            5. Log a human-readable summary.

        Returns:
            Tuple[List[Pair], List[Pair]]:
                (train_pairs, val_pairs) where each element is a list of
                (Path, Path, int) tuples ready for SiamesePairDataset.
        """
        positive_pairs = self._generate_positive_pairs()
        negative_pairs = self._generate_negative_pairs(target=len(positive_pairs))

        all_pairs = positive_pairs + negative_pairs
        self._rng.shuffle(all_pairs)

        self._train_pairs, self._val_pairs = self._split(all_pairs)

        logger.info(self._build_summary())
        return self._train_pairs, self._val_pairs

    def summary(self) -> str:
        """
        Return a human-readable summary of the last generate() call.

        Returns:
            str: Multi-line summary string.

        Raises:
            RuntimeError: If generate() has not been called yet.
        """
        if not self._train_pairs and not self._val_pairs:
            raise RuntimeError("Call generate() before requesting a summary.")
        return self._build_summary()

    # ------------------------------------------------------------------
    # Positive pair generation
    # ------------------------------------------------------------------

    def _generate_positive_pairs(self) -> List[Pair]:
        """
        Generate positive pairs (same identity, label=0) for every identity.

        For each identity:
            - Enumerate all unique (img_i, img_j) combinations via itertools.combinations.
            - If PAIRS_PER_IDENTITY <= unique_count → sample without replacement.
            - If PAIRS_PER_IDENTITY > unique_count  → sample with replacement,
              repeating some pairs. Logged as an INFO message so the user is aware.

        This sampling-with-replacement strategy is intentional: it allows identities
        with very few images (e.g. 2 images → 1 unique pair) to still contribute
        PAIRS_PER_IDENTITY training samples without crashing.

        Returns:
            List[Pair]: All positive pairs across all identities.
        """
        positive: List[Pair] = []
        target_per_identity = self._cfg.PAIRS_PER_IDENTITY

        for name, paths in self._identity_map.items():
            unique_combos = list(itertools.combinations(paths, 2))

            if not unique_combos:
                logger.warning(
                    f"Identity '{name}' has only 1 image — "
                    "cannot form any positive pairs. Skipping."
                )
                continue

            if len(unique_combos) >= target_per_identity:
                # Sample without replacement — no duplicate pairs
                selected = self._rng.sample(unique_combos, target_per_identity)
            else:
                # Sample with replacement — some pairs will repeat
                logger.info(
                    f"Identity '{name}' has {len(paths)} images "
                    f"({len(unique_combos)} unique pairs) but "
                    f"PAIRS_PER_IDENTITY={target_per_identity}. "
                    "Sampling with replacement to reach target."
                )
                selected = self._rng.choices(unique_combos, k=target_per_identity)

            for path_a, path_b in selected:
                positive.append((path_a, path_b, 0))

        logger.debug(f"Generated {len(positive)} positive pairs.")
        return positive

    # ------------------------------------------------------------------
    # Negative pair generation
    # ------------------------------------------------------------------

    def _generate_negative_pairs(self, target: int) -> List[Pair]:
        """
        Generate exactly `target` negative pairs (different identities, label=1).

        Strategy:
            Distribute the target evenly across all identities. For each
            identity i, sample `pairs_per_identity` anchor images from i,
            pairing each with a random image from a randomly chosen identity j ≠ i.

            If `target` is not evenly divisible by the number of identities,
            extra pairs are sampled from random identities until the target
            is exactly met.

        This design ensures every identity participates equally as an anchor
        in negative pairs, preventing the network from learning a trivial
        bias toward over-represented identities.

        Args:
            target: Exact number of negative pairs to generate. Must equal
                    the number of positive pairs for class balance.

        Returns:
            List[Pair]: Exactly `target` negative pairs.
        """
        identity_names = list(self._identity_map.keys())
        negative: List[Pair] = []

        n_identities = len(identity_names)
        base_per_identity, remainder = divmod(target, n_identities)

        for i, anchor_name in enumerate(identity_names):
            # Give one extra pair to the first `remainder` identities
            n_pairs = base_per_identity + (1 if i < remainder else 0)

            anchor_pool = self._identity_map[anchor_name]
            other_names = [n for n in identity_names if n != anchor_name]

            for _ in range(n_pairs):
                anchor_img  = self._rng.choice(anchor_pool)
                other_name  = self._rng.choice(other_names)
                other_img   = self._rng.choice(self._identity_map[other_name])
                negative.append((anchor_img, other_img, 1))

        logger.debug(f"Generated {len(negative)} negative pairs.")
        return negative

    # ------------------------------------------------------------------
    # Train / Val split
    # ------------------------------------------------------------------

    def _split(
        self,
        pairs: List[Pair],
    ) -> Tuple[List[Pair], List[Pair]]:
        """
        Perform a stratified train/validation split on a pre-shuffled pairs list.

        Because the input list is already shuffled and class-balanced (50/50),
        a simple index-based split preserves the balance in both subsets.
        No separate stratification library is needed.

        Args:
            pairs: Pre-shuffled list of (path1, path2, label) tuples.

        Returns:
            Tuple[List[Pair], List[Pair]]: (train_pairs, val_pairs).
        """
        split_idx = int(len(pairs) * self._cfg.TRAIN_SPLIT)
        train = pairs[:split_idx]
        val   = pairs[split_idx:]

        logger.debug(
            f"Split: {len(train)} train pairs, {len(val)} val pairs "
            f"({self._cfg.TRAIN_SPLIT:.0%} / {1 - self._cfg.TRAIN_SPLIT:.0%})"
        )
        return train, val

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _build_summary(self) -> str:
        """
        Construct a human-readable multi-line summary string.

        Returns:
            str: Formatted summary of the pair generation results.
        """
        total = len(self._train_pairs) + len(self._val_pairs)

        train_pos = sum(1 for _, _, l in self._train_pairs if l == 0)
        train_neg = sum(1 for _, _, l in self._train_pairs if l == 1)
        val_pos   = sum(1 for _, _, l in self._val_pairs   if l == 0)
        val_neg   = sum(1 for _, _, l in self._val_pairs   if l == 1)

        lines = [
            "",
            "┌─────────────────────────────────────────┐",
            "│         Pair Generation Summary          │",
            "├─────────────────────────────────────────┤",
            f"│  Identities       : {len(self._identity_map):<21}│",
            f"│  Pairs/Identity   : {self._cfg.PAIRS_PER_IDENTITY:<21}│",
            f"│  Total Pairs      : {total:<21}│",
            "├──────────────┬──────────┬───────────────┤",
            "│   Split      │ Positive │   Negative    │",
            "├──────────────┼──────────┼───────────────┤",
            f"│   Train      │ {train_pos:<8} │ {train_neg:<13} │",
            f"│   Val        │ {val_pos:<8} │ {val_neg:<13} │",
            f"│   Total      │ {train_pos+val_pos:<8} │ {train_neg+val_neg:<13} │",
            "└──────────────┴──────────┴───────────────┘",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def train_pairs(self) -> List[Pair]:
        """Training pairs from the last generate() call."""
        return self._train_pairs

    @property
    def val_pairs(self) -> List[Pair]:
        """Validation pairs from the last generate() call."""
        return self._val_pairs
