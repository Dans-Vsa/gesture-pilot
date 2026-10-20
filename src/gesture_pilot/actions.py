"""Actions triggered by gestures, and the backends that perform them.

Actions are written as short strings in the TOML config:

``key:<combo>``
    press a key or key combination, e.g. ``key:space``, ``key:ctrl+shift+t``,
    ``key:media_play_pause``, ``key:media_volume_up``
``click:<button>``
    mouse click, ``left`` (default), ``right`` or ``middle``; ``click:left:2``
    double-clicks
``scroll:<dy>`` / ``scroll:<dx>,<dy>``
    scroll by mouse-wheel steps (positive ``dy`` scrolls up)
``move`` / ``move:<speed>``
    continuous: while the gesture is held, the cursor follows the index finger
    tip (relative motion, ``speed`` pixels per frame width, default 1500)
``toggle``
    arm / disarm all other actions (the safety gesture)

Backends: :class:`DryRunBackend` only records what would happen;
:class:`PynputBackend` sends real input through pynput (imported lazily, so
tests and headless machines never need a display).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

#: Special keys accepted in ``key:`` actions (names of ``pynput.keyboard.Key``).
SPECIAL_KEYS = frozenset({
    "alt", "alt_l", "alt_r", "backspace", "caps_lock", "cmd", "ctrl", "ctrl_l", "ctrl_r",
    "delete", "down", "end", "enter", "esc", "home", "insert", "left", "menu",
    "page_down", "page_up", "pause", "print_screen", "right", "shift", "shift_l", "shift_r",
    "space", "tab", "up",
    "media_play_pause", "media_next", "media_previous", "media_volume_up",
    "media_volume_down", "media_volume_mute",
    *(f"f{i}" for i in range(1, 13)),
})
KEY_ALIASES = {"control": "ctrl", "escape": "esc", "return": "enter", "win": "cmd", "super": "cmd",
               "del": "delete", "pgup": "page_up", "pgdn": "page_down",
               "play_pause": "media_play_pause", "next": "media_next", "previous": "media_previous",
               "volume_up": "media_volume_up", "volume_down": "media_volume_down",
               "mute": "media_volume_mute"}
BUTTONS = ("left", "right", "middle")
KINDS = ("key", "click", "scroll", "move", "toggle")


class ActionError(ValueError):
    """Invalid action string."""


@dataclass(frozen=True)
class Action:
    kind: str
    keys: tuple[str, ...] = ()
    button: str = "left"
    count: int = 1
    dx: int = 0
    dy: int = 0
    speed: float = 1500.0
    spec: str = field(default="", compare=False)

    @property
    def continuous(self) -> bool:
        return self.kind == "move"

    def __str__(self) -> str:
        return self.spec or self.kind


def _parse_key(name: str) -> str:
    key = name.strip().lower()
    key = KEY_ALIASES.get(key, key)
    if len(key) == 1 and key.isprintable() and not key.isspace():
        return key
    if key in SPECIAL_KEYS:
        return key
    raise ActionError(f"unknown key {name!r}")


def _parse_int(text: str, what: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ActionError(f"{what} must be an integer, got {text!r}") from None


def parse_action(spec: str) -> Action:
    text = spec.strip()
    kind, _, arg = text.partition(":")
    kind = kind.lower()
    if kind == "key":
        if not arg:
            raise ActionError("key action needs keys, e.g. 'key:space' or 'key:ctrl+c'")
        keys = tuple(_parse_key(part) for part in arg.split("+"))
        return Action("key", keys=keys, spec=text)
    if kind == "click":
        parts = arg.split(":") if arg else []
        button = (parts[0] or "left").lower() if parts else "left"
        if button not in BUTTONS:
            raise ActionError(f"unknown mouse button {button!r}; choose from {', '.join(BUTTONS)}")
        count = _parse_int(parts[1], "click count") if len(parts) > 1 else 1
        if not 1 <= count <= 3:
            raise ActionError("click count must be 1, 2 or 3")
        return Action("click", button=button, count=count, spec=text)
    if kind == "scroll":
        parts = arg.split(",")
        if not arg or len(parts) > 2:
            raise ActionError("scroll action needs 'scroll:<dy>' or 'scroll:<dx>,<dy>'")
        values = [_parse_int(p.strip(), "scroll amount") for p in parts]
        dx, dy = (0, values[0]) if len(values) == 1 else (values[0], values[1])
        return Action("scroll", dx=dx, dy=dy, spec=text)
    if kind == "move":
        speed = 1500.0
        if arg:
            try:
                speed = float(arg)
            except ValueError:
                raise ActionError(f"move speed must be a number, got {arg!r}") from None
            if speed <= 0:
                raise ActionError("move speed must be positive")
        return Action("move", speed=speed, spec=text)
    if kind == "toggle":
        if arg:
            raise ActionError("toggle takes no arguments")
        return Action("toggle", spec=text)
    raise ActionError(f"unknown action {spec!r}; use one of: {', '.join(KINDS)}")


class ActionBackend(Protocol):
    def press(self, keys: tuple[str, ...]) -> None: ...
    def click(self, button: str, count: int) -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def move(self, dx: int, dy: int) -> None: ...


class DryRunBackend:
    """Records actions instead of performing them."""

    def __init__(self) -> None:
        self.log: list[str] = []

    def press(self, keys: tuple[str, ...]) -> None:
        self.log.append("press " + "+".join(keys))

    def click(self, button: str, count: int) -> None:
        self.log.append(f"click {button}" + (f" x{count}" if count > 1 else ""))

    def scroll(self, dx: int, dy: int) -> None:
        self.log.append(f"scroll {dx},{dy}")

    def move(self, dx: int, dy: int) -> None:
        self.log.append(f"move {dx},{dy}")


class PynputBackend:
    """Real keyboard and mouse input via pynput."""

    def __init__(self) -> None:
        try:
            from pynput import keyboard, mouse
        except Exception as exc:  # no display, missing backend, ...
            raise RuntimeError(f"pynput is not usable here: {exc}") from exc
        self._keyboard_mod = keyboard
        self._keyboard = keyboard.Controller()
        self._mouse = mouse.Controller()
        self._buttons = {"left": mouse.Button.left, "right": mouse.Button.right, "middle": mouse.Button.middle}

    def _key(self, name: str):
        if len(name) == 1:
            return name
        return getattr(self._keyboard_mod.Key, name)

    def press(self, keys: tuple[str, ...]) -> None:
        resolved = [self._key(k) for k in keys]
        for key in resolved:
            self._keyboard.press(key)
        for key in reversed(resolved):
            self._keyboard.release(key)

    def click(self, button: str, count: int) -> None:
        self._mouse.click(self._buttons[button], count)

    def scroll(self, dx: int, dy: int) -> None:
        self._mouse.scroll(dx, dy)

    def move(self, dx: int, dy: int) -> None:
        self._mouse.move(dx, dy)


def perform(action: Action, backend: ActionBackend) -> None:
    """Run a discrete action (``move`` and ``toggle`` are handled by the controller)."""
    if action.kind == "key":
        backend.press(action.keys)
    elif action.kind == "click":
        backend.click(action.button, action.count)
    elif action.kind == "scroll":
        backend.scroll(action.dx, action.dy)
    else:
        raise ActionError(f"{action.kind} is not a discrete action")
