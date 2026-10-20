"""TOML configuration: which gesture triggers which action.

Create a starting point with ``gesture-pilot init-config``; see
:data:`DEFAULT_CONFIG` for the format.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from gesture_pilot.actions import Action, ActionError, parse_action
from gesture_pilot.dataset import LABEL_PATTERN

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

HANDS = ("any", "Left", "Right")

DEFAULT_CONFIG = """\
# gesture-pilot: gesture -> action mapping
# Docs: https://github.com/Dans-Vsa/gesture-pilot#actions

[settings]
# Actions start disabled; hold the "toggle" gesture to arm / disarm them.
start_armed = false
# Frames a gesture must be held before it fires, and seconds between events.
stable_frames = 5
cooldown = 0.6
# Which hand may trigger actions: "any", "Left" or "Right".
hand = "any"

[actions]
ok = "toggle"                        # safety gesture: arm / disarm everything else
open_palm = "key:media_play_pause"
thumbs_up = "key:media_volume_up"
thumbs_down = "key:media_volume_down"
peace = "key:media_next"
point = "move"                       # cursor follows the index finger while held
pinch = "click:left"
# fist = "scroll:-3"                 # fist is a good "rest" pose, so it is unmapped by default
# Custom gestures from `gesture-pilot train` work too, e.g.:
# rock_on = "key:ctrl+shift+t"
"""


class ConfigError(Exception):
    """Invalid configuration file."""


@dataclass(frozen=True)
class Config:
    actions: dict[str, Action]
    start_armed: bool = False
    stable_frames: int = 5
    cooldown: float = 0.6
    hand: str = "any"

    @property
    def toggle_gesture(self) -> str | None:
        for gesture, action in self.actions.items():
            if action.kind == "toggle":
                return gesture
        return None


def parse_config(text: str, source: str = "<config>") -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{source}: invalid TOML: {exc}") from None

    unknown = set(data) - {"settings", "actions"}
    if unknown:
        raise ConfigError(f"{source}: unknown section(s): {', '.join(sorted(unknown))}")
    settings = data.get("settings", {})
    raw_actions = data.get("actions", {})
    if not isinstance(settings, dict) or not isinstance(raw_actions, dict):
        raise ConfigError(f"{source}: [settings] and [actions] must be tables")

    actions: dict[str, Action] = {}
    for gesture, spec in raw_actions.items():
        if not LABEL_PATTERN.match(gesture) or gesture == "none":
            raise ConfigError(f"{source}: invalid gesture name {gesture!r}")
        if not isinstance(spec, str):
            raise ConfigError(f"{source}: action for {gesture!r} must be a string")
        try:
            actions[gesture] = parse_action(spec)
        except ActionError as exc:
            raise ConfigError(f"{source}: {gesture}: {exc}") from None
    if not actions:
        raise ConfigError(f"{source}: no actions configured")
    toggles = [g for g, a in actions.items() if a.kind == "toggle"]
    if len(toggles) > 1:
        raise ConfigError(f"{source}: only one toggle gesture allowed, got {', '.join(toggles)}")

    known = {"start_armed", "stable_frames", "cooldown", "hand"}
    extra = set(settings) - known
    if extra:
        raise ConfigError(f"{source}: unknown setting(s): {', '.join(sorted(extra))}")
    start_armed = settings.get("start_armed", False)
    stable_frames = settings.get("stable_frames", 5)
    cooldown = settings.get("cooldown", 0.6)
    hand = settings.get("hand", "any")
    if not isinstance(start_armed, bool):
        raise ConfigError(f"{source}: start_armed must be true or false")
    if not isinstance(stable_frames, int) or isinstance(stable_frames, bool) or stable_frames < 1:
        raise ConfigError(f"{source}: stable_frames must be a positive integer")
    if not isinstance(cooldown, (int, float)) or isinstance(cooldown, bool) or cooldown < 0:
        raise ConfigError(f"{source}: cooldown must be a number >= 0")
    if hand not in HANDS:
        raise ConfigError(f"{source}: hand must be one of {', '.join(HANDS)}")
    if not toggles:
        start_armed = True  # without a safety gesture there is no way to arm later
    return Config(actions, start_armed, stable_frames, float(cooldown), hand)


def load_config(path: Path | None) -> Config:
    """Load ``path``, or the built-in default mapping when ``path`` is ``None``."""
    if path is None:
        return parse_config(DEFAULT_CONFIG, "<default config>")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config not found: {path} (create one with `gesture-pilot init-config`)") from None
    return parse_config(text, str(path))
