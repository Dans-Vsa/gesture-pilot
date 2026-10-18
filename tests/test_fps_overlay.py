import numpy as np
import pytest

from gesture_pilot.fps import FPSCounter
from gesture_pilot.overlay import draw_fps, draw_label


class FakeClock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def test_constant_rate_converges() -> None:
    counter = FPSCounter(clock=FakeClock(1 / 30))
    for _ in range(50):
        counter.tick()
    assert counter.fps == pytest.approx(30.0)
    assert counter.average == pytest.approx(30.0)
    assert counter.frames == 50


def test_first_tick_has_no_rate() -> None:
    counter = FPSCounter(clock=FakeClock(0.1))
    assert counter.tick() == 0.0
    assert counter.average == 0.0


def test_smoothing_dampens_spikes() -> None:
    clock = FakeClock(0.1)  # 10 FPS
    counter = FPSCounter(smoothing=0.9, clock=clock)
    for _ in range(20):
        counter.tick()
    clock.step = 0.01  # the clock already advanced one slow step; the next gap is fast
    counter.tick()  # still a 10 FPS interval
    counter.tick()  # one very fast frame (100 FPS)
    assert 10 < counter.fps < 25


def test_invalid_smoothing() -> None:
    with pytest.raises(ValueError):
        FPSCounter(smoothing=1.0)


def test_draw_fps_changes_top_left_only() -> None:
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    draw_fps(frame, 29.7)
    assert frame[:40, :120].any()
    assert not frame[80:, 150:].any()


def test_draw_label_returns_same_frame() -> None:
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    assert draw_label(frame, "hi", background=(10, 20, 30)) is frame
    assert tuple(frame[12, 12]) == (10, 20, 30)
