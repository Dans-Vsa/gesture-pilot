"""Frame-rate measurement with exponential smoothing."""

from __future__ import annotations

import time
from typing import Callable


class FPSCounter:
    """Measure frames per second.

    ``tick()`` is called once per processed frame. The instantaneous rate is
    smoothed with an exponential moving average so the on-screen number does
    not flicker. ``clock`` is injectable for deterministic tests.
    """

    def __init__(self, smoothing: float = 0.9, clock: Callable[[], float] = time.perf_counter) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing
        self._clock = clock
        self._last: float | None = None
        self._start: float | None = None
        self.fps = 0.0
        self.frames = 0

    def tick(self) -> float:
        now = self._clock()
        if self._start is None:
            self._start = now
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                if self.fps == 0.0:
                    self.fps = instant
                else:
                    self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant
        self._last = now
        self.frames += 1
        return self.fps

    @property
    def elapsed(self) -> float:
        if self._start is None or self._last is None:
            return 0.0
        return self._last - self._start

    @property
    def average(self) -> float:
        """Average FPS over the whole run."""
        if self.frames < 2 or self.elapsed <= 0:
            return 0.0
        return (self.frames - 1) / self.elapsed
