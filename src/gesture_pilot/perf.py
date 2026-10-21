"""Performance helpers: frame skipping and downscaling before detection.

Hand detection is by far the most expensive step of the loop. Two cheap knobs
trade a little accuracy for a lot of speed on slow machines:

* **frame skipping** — run the detector on every Nth frame only and reuse the
  last detected hands in between (the overlay and actions keep running at the
  full camera frame rate);
* **detection width** — downscale frames before they are handed to the
  detector. MediaPipe returns landmarks normalised to ``[0, 1]``, so they map
  back onto the full-size frame without any conversion.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Smallest width accepted for ``--detect-width``; smaller frames make the
#: palm detector miss hands that are not very close to the camera.
MIN_DETECT_WIDTH = 160


class FrameSkipper:
    """Decide which frames go through the (expensive) hand detector."""

    def __init__(self, every: int = 1) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        self.every = every
        self.processed = 0
        self.skipped = 0

    def should_process(self, index: int) -> bool:
        """``True`` for frame ``index`` when it should be detected (0, N, 2N, ...)."""
        process = index % self.every == 0
        if process:
            self.processed += 1
        else:
            self.skipped += 1
        return process


def detection_size(width: int, height: int, max_width: int | None) -> tuple[int, int]:
    """Size of a ``width`` x ``height`` frame after limiting it to ``max_width``.

    Frames that are already narrow enough keep their size; the aspect ratio is
    always preserved.
    """
    if max_width is None or width <= max_width:
        return width, height
    if max_width < MIN_DETECT_WIDTH:
        raise ValueError(f"max_width must be >= {MIN_DETECT_WIDTH}")
    scale = max_width / width
    return max_width, max(1, round(height * scale))


def prepare_for_detection(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    """Return ``frame`` downscaled to at most ``max_width`` pixels wide."""
    height, width = frame.shape[:2]
    size = detection_size(width, height, max_width)
    if size == (width, height):
        return frame
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
