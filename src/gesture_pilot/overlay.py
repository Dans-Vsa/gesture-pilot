"""Drawing helpers for the on-screen overlay."""

from __future__ import annotations

import cv2
import numpy as np

from gesture_pilot.landmarks import HAND_CONNECTIONS, Hand

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


BONE_COLOR = (230, 230, 230)
JOINT_COLOR = (0, 140, 255)
TIP_COLOR = (0, 220, 255)
TIPS = (4, 8, 12, 16, 20)


def draw_hand(
    frame: np.ndarray,
    hand: Hand,
    *,
    label: str | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw the 21-point skeleton of ``hand`` (and an optional label). In place."""
    height, width = frame.shape[:2]
    px = hand.pixels(width, height)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(map(int, px[a])), tuple(map(int, px[b])), BONE_COLOR, thickness, cv2.LINE_AA)
    radius = max(2, thickness + 1)
    for i, (x, y) in enumerate(px):
        color = TIP_COLOR if i in TIPS else JOINT_COLOR
        cv2.circle(frame, (int(x), int(y)), radius + (1 if i in TIPS else 0), color, -1, cv2.LINE_AA)
    if label:
        x0, y0, _, _ = hand.bbox()
        origin = (max(0, int(x0 * width)), max(0, int(y0 * height) - 30))
        draw_label(frame, label, origin, scale=0.55, color=(255, 255, 255), background=(90, 40, 0))
    return frame
