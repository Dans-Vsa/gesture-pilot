"""MediaPipe HandLandmarker wrapper (VIDEO running mode)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from gesture_pilot.landmarks import Hand
from gesture_pilot.model import default_model_path


class TrackerError(RuntimeError):
    """Raised when the hand tracker cannot be created."""


class HandTracker:
    """Detect hands in BGR frames and return :class:`Hand` objects.

    The VIDEO running mode tracks hands between frames (faster and steadier
    than detecting on every frame) but needs strictly increasing timestamps;
    :meth:`detect` enforces that, so callers can pass wall-clock or
    frame-based times without worrying about duplicates.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        path = Path(model_path) if model_path else default_model_path()
        if not path.is_file():
            raise TrackerError(f"model not found: {path} (run `gesture-pilot download-model` first)")

        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except (RuntimeError, ValueError) as exc:
            raise TrackerError(f"could not load model {path}: {exc}") from exc
        self.model_path = path
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        import mediapipe as mp

        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, ts)
        return hands_from_result(result)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def hands_from_result(result: object) -> list[Hand]:
    """Convert a MediaPipe ``HandLandmarkerResult`` into :class:`Hand` objects."""
    hands: list[Hand] = []
    landmarks = getattr(result, "hand_landmarks", None) or []
    handedness = getattr(result, "handedness", None) or []
    for i, marks in enumerate(landmarks):
        points = np.array([[lm.x, lm.y, lm.z] for lm in marks], dtype=np.float32)
        label, score = "Right", 1.0
        if i < len(handedness) and handedness[i]:
            category = handedness[i][0]
            label = category.category_name if category.category_name in ("Left", "Right") else label
            score = float(category.score)
        hands.append(Hand(points, handedness=label, score=score))
    return hands
