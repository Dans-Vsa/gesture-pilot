from pathlib import Path

import cv2
import numpy as np
import pytest

from gesture_pilot.sources import (
    ImageSource,
    SourceError,
    TestPatternSource,
    VideoSource,
    open_source,
)


def _write_video(path: Path, frames: int = 6, size: tuple[int, int] = (64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, size)
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_test_pattern_is_finite_and_animated() -> None:
    src = TestPatternSource(80, 60, frames=3)
    frames = list(src)
    assert len(frames) == 3
    assert all(f.shape == (60, 80, 3) and f.dtype == np.uint8 for f in frames)
    assert not np.array_equal(frames[0], frames[2])


def test_open_source_test_pattern_respects_size_and_limit() -> None:
    with open_source("test", width=32, height=24, max_frames=2) as src:
        frames = list(src)
    assert len(frames) == 2
    assert frames[0].shape == (24, 32, 3)


def test_image_source_repeats_copies(tmp_path: Path) -> None:
    img = tmp_path / "hand.png"
    cv2.imwrite(str(img), np.full((10, 12, 3), 200, dtype=np.uint8))
    src = open_source(str(img), max_frames=3)
    assert isinstance(src, ImageSource)
    frames = list(src)
    assert len(frames) == 3
    frames[0][:] = 0  # mutating one frame must not affect the others
    assert frames[1].max() == 200


def test_unreadable_image_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(SourceError):
        open_source(str(bad))


def test_video_source_reads_all_frames(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "clip.avi", frames=6)
    with open_source(str(video)) as src:
        assert isinstance(src, VideoSource)
        assert src.fps == pytest.approx(15.0)
        frames = list(src)
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        open_source(str(tmp_path / "missing.mp4"))
