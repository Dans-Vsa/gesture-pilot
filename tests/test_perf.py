"""Frame skipping and detection downscaling."""

import numpy as np
import pytest

from gesture_pilot.perf import MIN_DETECT_WIDTH, FrameSkipper, detection_size, prepare_for_detection


def test_frame_skipper_every_frame_by_default() -> None:
    skipper = FrameSkipper()
    assert all(skipper.should_process(i) for i in range(5))
    assert (skipper.processed, skipper.skipped) == (5, 0)


def test_frame_skipper_every_third_frame() -> None:
    skipper = FrameSkipper(3)
    picked = [i for i in range(10) if skipper.should_process(i)]
    assert picked == [0, 3, 6, 9]
    assert (skipper.processed, skipper.skipped) == (4, 6)


def test_frame_skipper_rejects_zero() -> None:
    with pytest.raises(ValueError):
        FrameSkipper(0)


@pytest.mark.parametrize(
    ("size", "max_width", "expected"),
    [
        ((640, 480), None, (640, 480)),
        ((640, 480), 640, (640, 480)),
        ((640, 480), 1280, (640, 480)),
        ((640, 480), 320, (320, 240)),
        ((1280, 720), 480, (480, 270)),
        ((1920, 1080), 333, (333, 187)),
    ],
)
def test_detection_size_keeps_aspect_ratio(
    size: tuple[int, int], max_width: int | None, expected: tuple[int, int]
) -> None:
    assert detection_size(*size, max_width) == expected


def test_detection_size_rejects_tiny_width() -> None:
    with pytest.raises(ValueError):
        detection_size(640, 480, MIN_DETECT_WIDTH - 1)


def test_prepare_for_detection_downscales_copy() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    small = prepare_for_detection(frame, 320)
    assert small.shape == (240, 320, 3)
    assert small.dtype == np.uint8
    assert int(small[100, 100, 0]) == 200
    assert frame.shape == (480, 640, 3)  # original untouched


def test_prepare_for_detection_returns_same_frame_when_small_enough() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert prepare_for_detection(frame, 320) is frame
    assert prepare_for_detection(frame, None) is frame
