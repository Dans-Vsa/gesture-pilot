"""Drawing helpers for the on-screen overlay."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int] = (10, 10),
    *,
    scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    background: tuple[int, int, int] = (0, 0, 0),
    padding: int = 6,
) -> np.ndarray:
    """Draw ``text`` on a filled box whose top-left corner is ``origin``. In place."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, 1)
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y),
        (x + tw + 2 * padding, y + th + baseline + 2 * padding),
        background,
        -1,
    )
    cv2.putText(frame, text, (x + padding, y + padding + th), FONT, scale, color, 1, cv2.LINE_AA)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw an FPS badge in the top-left corner. In place."""
    return draw_label(frame, f"FPS {fps:5.1f}", (10, 10), color=(80, 255, 80))
