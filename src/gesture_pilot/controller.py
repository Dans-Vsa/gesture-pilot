"""Turn gesture events into actions, with a safety toggle.

Discrete actions (keys, clicks, scrolling) fire on debounced
:class:`~gesture_pilot.smoothing.GestureEvent`\\ s. The continuous ``move``
action is driven frame by frame: while a hand shows the move gesture, the
cursor follows the motion of its index fingertip.
"""

from __future__ import annotations

from gesture_pilot.actions import ActionBackend, perform
from gesture_pilot.config import Config
from gesture_pilot.landmarks import INDEX_TIP, Hand
from gesture_pilot.smoothing import GestureEvent

#: Fingertip motion below this (fraction of the frame) is treated as jitter.
MOVE_DEADZONE = 0.002


class GestureController:
    def __init__(self, config: Config, backend: ActionBackend) -> None:
        self.config = config
        self.backend = backend
        self.armed = config.start_armed
        self._last_tip: dict[str, tuple[float, float]] = {}

    def _hand_allowed(self, handedness: str) -> bool:
        return self.config.hand in ("any", handedness)

    def handle_event(self, event: GestureEvent) -> str | None:
        """Run the action mapped to ``event``; return a log message or ``None``."""
        if not self._hand_allowed(event.handedness):
            return None
        action = self.config.actions.get(event.gesture)
        if action is None or action.continuous:
            return None
        if action.kind == "toggle":
            self.armed = not self.armed
            self._last_tip.clear()
            return "armed" if self.armed else "disarmed"
        if not self.armed:
            return f"{event.gesture} -> {action} (ignored: disarmed)"
        perform(action, self.backend)
        return f"{event.gesture} -> {action}"

    def update_frame(self, hands: list[Hand], labels: dict[str, str]) -> None:
        """Drive continuous actions from this frame's hands and per-frame labels."""
        moving: set[str] = set()
        if self.armed:
            for hand in hands:
                key = hand.handedness
                if key in moving or not self._hand_allowed(key):
                    continue
                action = self.config.actions.get(labels.get(key, ""))
                if action is None or not action.continuous:
                    continue
                moving.add(key)
                tip = (float(hand.points[INDEX_TIP, 0]), float(hand.points[INDEX_TIP, 1]))
                last = self._last_tip.get(key)
                if last is None:
                    self._last_tip[key] = tip
                    continue
                dx, dy = tip[0] - last[0], tip[1] - last[1]
                if abs(dx) < MOVE_DEADZONE and abs(dy) < MOVE_DEADZONE:
                    continue
                self._last_tip[key] = tip
                px, py = round(dx * action.speed), round(dy * action.speed)
                if px or py:
                    self.backend.move(px, py)
        for key in list(self._last_tip):
            if key not in moving:
                del self._last_tip[key]
