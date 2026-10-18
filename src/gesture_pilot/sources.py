"""Frame sources: webcam, video file, still image and a synthetic test pattern.

Every source yields BGR ``numpy`` frames, so the rest of the pipeline does not
care where pixels come from. This keeps the project testable without a camera:
tests and CI use video files, images or the ``test`` pattern.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TEST_PATTERN = "test"


class SourceError(RuntimeError):
    """Raised when a frame source cannot be opened."""


class FrameSource(ABC):
    """Iterable of BGR frames. Use as a context manager to release resources."""

    name: str = "source"
    #: Frames per second the source runs at, if known (video files, cameras).
    fps: float | None = None
    #: True when timestamps should follow the wall clock (live sources).
    live: bool = False

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next frame or ``None`` when the source is exhausted."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the underlying resource."""

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> FrameSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _CaptureSource(FrameSource):
    def __init__(self, capture: cv2.VideoCapture, name: str) -> None:
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"could not open {name}")
        self._cap = capture
        self.name = name
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and fps > 0 else None

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class CameraSource(_CaptureSource):
    """A webcam addressed by its index (0 = default camera)."""

    live = True

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        capture = cv2.VideoCapture(index)
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        super().__init__(capture, f"camera {index}")


class VideoSource(_CaptureSource):
    """A video file, played frame by frame."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise SourceError(f"video file not found: {path}")
        super().__init__(cv2.VideoCapture(str(path)), f"video {path.name}")


class ImageSource(FrameSource):
    """A still image, repeated ``repeat`` times (default: once)."""

    def __init__(self, path: str | Path, repeat: int = 1) -> None:
        path = Path(path)
        frame = cv2.imread(str(path))
        if frame is None:
            raise SourceError(f"could not read image: {path}")
        self._frame = frame
        self._remaining = max(1, repeat)
        self.name = f"image {path.name}"

    def read(self) -> np.ndarray | None:
        if self._remaining <= 0:
            return None
        self._remaining -= 1
        return self._frame.copy()


class TestPatternSource(FrameSource):
    """Animated synthetic frames, handy when no camera is available."""

    __test__ = False  # not a pytest test class

    def __init__(self, width: int = 640, height: int = 480, frames: int | None = None) -> None:
        self.width = width
        self.height = height
        self._limit = frames
        self._index = 0
        self.name = "test pattern"
        self.fps = 30.0

    def read(self) -> np.ndarray | None:
        if self._limit is not None and self._index >= self._limit:
            return None
        t = self._index
        self._index += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Horizontal colour gradient that slowly shifts over time.
        ramp = np.linspace(0, 255, self.width, dtype=np.float32)
        frame[:, :, 0] = ((ramp + t * 4) % 256).astype(np.uint8)
        frame[:, :, 1] = 60
        frame[:, :, 2] = (255 - ramp).astype(np.uint8)
        # A white dot moving on a circle, like a fingertip.
        cx = int(self.width / 2 + math.cos(t / 15) * self.width / 4)
        cy = int(self.height / 2 + math.sin(t / 15) * self.height / 4)
        cv2.circle(frame, (cx, cy), max(4, self.width // 40), (255, 255, 255), -1)
        return frame


def open_source(
    spec: str | int,
    *,
    width: int | None = None,
    height: int | None = None,
    max_frames: int | None = None,
) -> FrameSource:
    """Open a source from a CLI-style spec.

    * ``"0"``, ``"1"``, ... or an int → webcam index
    * ``"test"`` → synthetic test pattern
    * a path ending in an image suffix → still image
    * any other path → video file
    """
    if isinstance(spec, int) or (isinstance(spec, str) and spec.strip().isdigit()):
        return CameraSource(int(spec), width=width, height=height)
    if spec == TEST_PATTERN:
        return TestPatternSource(width or 640, height or 480, frames=max_frames)
    path = Path(spec)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return ImageSource(path, repeat=max_frames or 1)
    return VideoSource(path)
