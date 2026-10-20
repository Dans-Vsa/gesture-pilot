"""Labelled gesture samples stored as CSV.

Each row is one hand: ``label, handedness, f0 … f62`` where ``f*`` is the
63-value :func:`gesture_pilot.landmarks.feature_vector` (normalised, so the
data does not depend on where the hand was in the frame or which hand it was).
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_pilot.landmarks import NUM_LANDMARKS, Hand, feature_vector

FEATURE_SIZE = NUM_LANDMARKS * 3
HEADER = ["label", "handedness", *(f"f{i}" for i in range(FEATURE_SIZE))]
LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class DatasetError(Exception):
    """Invalid or unreadable dataset file."""


@dataclass(frozen=True)
class Dataset:
    features: np.ndarray  # (n, 63) float32
    labels: np.ndarray  # (n,) str

    def __len__(self) -> int:
        return len(self.labels)

    def counts(self) -> dict[str, int]:
        return dict(sorted(Counter(self.labels.tolist()).items()))


def validate_label(label: str) -> str:
    """Labels are lower-case identifiers such as ``rock_on`` or ``swipe-left``."""
    if not LABEL_PATTERN.match(label) or label == "none":
        raise DatasetError(
            f"invalid label {label!r}: use lower-case letters, digits, '_' or '-' (and not 'none')"
        )
    return label


def sample_row(label: str, hand: Hand) -> list[str]:
    values = feature_vector(hand)
    return [label, hand.handedness, *(f"{v:.5f}" for v in values)]


def append_samples(path: Path, rows: Iterable[list[str]]) -> int:
    """Append rows to ``path`` (writing the header for a new file); return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open(newline="", encoding="utf-8") as fh:
            first = next(csv.reader(fh), None)
        if first != HEADER:
            raise DatasetError(f"{path} is not a gesture-pilot dataset (unexpected header)")
    written = 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)
            written += 1
    return written


def load_dataset(path: Path) -> Dataset:
    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    features: list[list[float]] = []
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != HEADER:
            raise DatasetError(f"{path} is not a gesture-pilot dataset (unexpected header)")
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(HEADER):
                raise DatasetError(f"{path}:{line_no}: expected {len(HEADER)} columns, got {len(row)}")
            try:
                features.append([float(v) for v in row[2:]])
            except ValueError:
                raise DatasetError(f"{path}:{line_no}: non-numeric feature value") from None
            labels.append(row[0])
    if not labels:
        raise DatasetError(f"{path} contains no samples")
    return Dataset(np.asarray(features, dtype=np.float32), np.asarray(labels))


def synthetic_samples(
    per_class: int = 60,
    *,
    poses: Iterable[str] | None = None,
    seed: int | None = 0,
) -> list[list[str]]:
    """Rows for the built-in poses from :mod:`gesture_pilot.synthetic`.

    Hands get random handedness, position, size, a little rotation and noise,
    so a classifier trained on them generalises over the same variations a real
    recording has. Handy for demos and tests without a camera.
    """
    from gesture_pilot.synthetic import POSES, make_pose

    rng = np.random.default_rng(seed)
    rows: list[list[str]] = []
    for name in poses or POSES:
        for _ in range(per_class):
            hand = make_pose(
                name,
                handedness=str(rng.choice(["Left", "Right"])),
                rotation=float(rng.uniform(-15, 15)),
                center=(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.4, 0.7))),
                scale=float(rng.uniform(0.1, 0.25)),
                jitter=0.03,
                rng=rng,
            )
            rows.append(sample_row(name, hand))
    return rows
