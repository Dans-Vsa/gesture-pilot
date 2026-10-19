"""Temporal smoothing: steady landmarks and debounced gesture events.

Per-frame recognition flickers (a finger half-bent for one frame turns
``open_palm`` into ``none``). Two small helpers fix that:

* :class:`LandmarkSmoother` — exponential moving average of landmark
  positions, one track per hand (keyed by handedness);
* :class:`GestureStabilizer` — a gesture must be seen for ``stable_frames``
  consecutive frames before it fires, the same gesture does not fire again
  until it was released, and nothing fires during the ``cooldown`` after an
  event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesture_pilot.gestures import NONE
from gesture_pilot.landmarks import Hand


class LandmarkSmoother:
    """Exponential moving average of hand landmarks.

    ``alpha`` is the weight of the newest frame (1.0 = no smoothing). A hand
    that disappears resets its track, so it does not glide in from its old
    position when it comes back.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: dict[str, np.ndarray] = {}

    def update(self, hands: list[Hand]) -> list[Hand]:
        seen: set[str] = set()
        smoothed: list[Hand] = []
        for hand in hands:
            key = hand.handedness
            if key in seen:  # two hands with the same label: do not mix tracks
                smoothed.append(hand)
                continue
            seen.add(key)
            prev = self._state.get(key)
            points = hand.points if prev is None else self.alpha * hand.points + (1 - self.alpha) * prev
            self._state[key] = points
            smoothed.append(Hand(points, hand.handedness, hand.score, dict(hand.extra)))
        for key in list(self._state):
            if key not in seen:
                del self._state[key]
        return smoothed

    def reset(self) -> None:
        self._state.clear()


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    handedness: str
    timestamp: float  # seconds


class GestureStabilizer:
    """Debounce per-frame gesture labels for one hand."""

    def __init__(self, stable_frames: int = 5, cooldown: float = 0.5) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self._candidate = NONE
        self._count = 0
        self._active = NONE  # last gesture that fired and is still held
        self._last_fire: float | None = None

    @property
    def active(self) -> str:
        """Gesture currently held (fired and not yet released), or ``none``."""
        return self._active

    def update(self, gesture: str, timestamp: float, handedness: str = "Right") -> GestureEvent | None:
        """Feed one frame's label; return an event when a gesture fires."""
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1

        if self._count < self.stable_frames:
            return None
        if gesture == NONE:
            self._active = NONE
            return None
        if gesture == self._active:
            return None
        if self._last_fire is not None and timestamp - self._last_fire < self.cooldown:
            return None
        self._active = gesture
        self._last_fire = timestamp
        return GestureEvent(gesture, handedness, timestamp)

    def reset(self) -> None:
        self._candidate = NONE
        self._count = 0
        self._active = NONE


class MultiHandStabilizer:
    """One :class:`GestureStabilizer` per handedness."""

    def __init__(self, stable_frames: int = 5, cooldown: float = 0.5) -> None:
        self.stable_frames = stable_frames
        self.cooldown = cooldown
        self._hands: dict[str, GestureStabilizer] = {}

    def update(self, labels: dict[str, str], timestamp: float) -> list[GestureEvent]:
        """``labels`` maps handedness → gesture for this frame.

        Hands missing from ``labels`` are treated as showing ``none``.
        """
        events: list[GestureEvent] = []
        for hand in sorted(set(labels) | set(self._hands)):
            stab = self._hands.setdefault(hand, GestureStabilizer(self.stable_frames, self.cooldown))
            event = stab.update(labels.get(hand, NONE), timestamp, hand)
            if event is not None:
                events.append(event)
        return events

    def active(self, handedness: str) -> str:
        stab = self._hands.get(handedness)
        return stab.active if stab else NONE
