"""Hand landmark data structures and normalisation.

MediaPipe returns 21 landmarks per hand in normalised image coordinates
(``x`` and ``y`` in ``[0, 1]``, ``y`` pointing down, ``z`` relative depth with
the wrist at roughly 0). Everything downstream (gesture rules, the trainable
classifier) works on :class:`Hand` objects, so it can be tested with synthetic
arrays instead of a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_LANDMARKS = 21

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

#: Landmark indices of each finger, from base to tip.
FINGERS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

#: Bones of the hand skeleton (pairs of landmark indices), same as MediaPipe.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

HANDEDNESS = ("Left", "Right")


@dataclass
class Hand:
    """One detected hand.

    ``points`` is a ``(21, 3)`` float array in normalised image coordinates.
    ``handedness`` is ``"Left"`` or ``"Right"`` as seen by the model.
    """

    points: np.ndarray
    handedness: str = "Right"
    score: float = 1.0
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.shape == (NUM_LANDMARKS, 2):
            points = np.column_stack([points, np.zeros(NUM_LANDMARKS, dtype=np.float32)])
        if points.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {points.shape}")
        if self.handedness not in HANDEDNESS:
            raise ValueError(f"handedness must be 'Left' or 'Right', got {self.handedness!r}")
        self.points = points

    def pixels(self, width: int, height: int) -> np.ndarray:
        """Landmarks as integer ``(x, y)`` pixel coordinates, shape ``(21, 2)``."""
        xy = self.points[:, :2] * np.array([width, height], dtype=np.float32)
        return np.rint(xy).astype(np.int32)

    def palm_size(self) -> float:
        """Distance wrist → middle-finger MCP in image space (2-D)."""
        return float(np.linalg.norm(self.points[MIDDLE_MCP, :2] - self.points[WRIST, :2]))

    def bbox(self) -> tuple[float, float, float, float]:
        """Normalised bounding box ``(x_min, y_min, x_max, y_max)``."""
        xy = self.points[:, :2]
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


def normalize(hand: Hand, *, mirror_left: bool = True) -> np.ndarray:
    """Translation-, scale- and handedness-invariant landmarks.

    * the wrist becomes the origin,
    * coordinates are divided by the palm size (wrist → middle MCP),
    * ``y`` is flipped so that "up" in the image is positive,
    * left hands are mirrored (``x → -x``) so they look like right hands.

    Rotation is kept on purpose: thumbs up and thumbs down only differ by
    orientation. Returns a ``(21, 3)`` float32 array.
    """
    pts = hand.points.astype(np.float32) - hand.points[WRIST].astype(np.float32)
    scale = float(np.linalg.norm(pts[MIDDLE_MCP, :2]))
    if scale < 1e-6:
        raise ValueError("degenerate hand: palm size is zero")
    pts = pts / scale
    pts[:, 1] *= -1.0
    if mirror_left and hand.handedness == "Left":
        pts[:, 0] *= -1.0
    return pts


def feature_vector(hand: Hand) -> np.ndarray:
    """Flattened normalised landmarks (63 values) — input for classifiers."""
    return normalize(hand).reshape(-1)
