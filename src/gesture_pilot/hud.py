"""On-screen heads-up display: mode, gestures, armed state and last action.

:class:`HudState` collects what the loop knows about the current frame and
:meth:`HudState.lines` turns it into plain text lines (easy to test);
:func:`draw_hud` renders them in a translucent panel in the bottom-left corner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from gesture_pilot.overlay import FONT

#: Seconds the last action stays highlighted on the HUD.
ACTION_HIGHLIGHT = 1.5

WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 220, 80)
AMBER = (0, 190, 255)


@dataclass
class HudState:
    mode: str
    gestures: dict[str, str] = field(default_factory=dict)
    armed: bool | None = None  # None: no action controller (track / recognize)
    last_action: str | None = None
    last_action_at: float = 0.0
    process_every: int = 1
    detect_size: tuple[int, int] | None = None
    dry_run: bool = False

    def set_action(self, message: str, now: float) -> None:
        self.last_action = message
        self.last_action_at = now

    def lines(self, now: float) -> list[tuple[str, tuple[int, int, int]]]:
        """Text lines and their BGR colours, top to bottom."""
        mode = self.mode + (" (dry run)" if self.dry_run else "")
        perf = []
        if self.process_every > 1:
            perf.append(f"detect 1/{self.process_every}")
        if self.detect_size is not None:
            perf.append(f"{self.detect_size[0]}x{self.detect_size[1]}")
        out = [(f"mode: {mode}" + (f"  [{', '.join(perf)}]" if perf else ""), WHITE)]
        if self.armed is not None:
            out.append(("ARMED" if self.armed else "disarmed", GREEN if self.armed else GREY))
        if self.gestures:
            text = "  ".join(f"{hand}: {name}" for hand, name in sorted(self.gestures.items()))
            out.append((text, WHITE))
        if self.last_action:
            fresh = now - self.last_action_at < ACTION_HIGHLIGHT
            out.append((f"last: {self.last_action}", AMBER if fresh else GREY))
        return out


def draw_hud(
    frame: np.ndarray,
    state: HudState,
    now: float,
    *,
    scale: float = 0.5,
    padding: int = 8,
    alpha: float = 0.55,
) -> np.ndarray:
    """Draw ``state`` in a translucent panel at the bottom-left. In place."""
    lines = state.lines(now)
    height, width = frame.shape[:2]
    sizes = [cv2.getTextSize(text, FONT, scale, 1)[0] for text, _ in lines]
    line_h = max(h for _, h in sizes) + 8
    panel_w = min(width, max(w for w, _ in sizes) + 2 * padding)
    panel_h = min(height, line_h * len(lines) + 2 * padding - 8)
    x0, y0 = 0, height - panel_h

    region = frame[y0:height, x0:x0 + panel_w]
    region[:] = (region * (1 - alpha)).astype(frame.dtype)
    for i, (text, color) in enumerate(lines):
        baseline_y = y0 + padding + (i + 1) * line_h - 8
        cv2.putText(frame, text, (x0 + padding, baseline_y), FONT, scale, color, 1, cv2.LINE_AA)
    return frame
