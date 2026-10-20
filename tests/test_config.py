from pathlib import Path

import pytest

from gesture_pilot.config import DEFAULT_CONFIG, ConfigError, load_config, parse_config
from gesture_pilot.gestures import GESTURES


def test_default_config_is_valid_and_safe() -> None:
    config = load_config(None)
    assert config == parse_config(DEFAULT_CONFIG)
    assert config.start_armed is False
    assert config.toggle_gesture == "ok"
    assert set(config.actions) <= set(GESTURES)
    assert config.actions["point"].kind == "move"
    assert config.hand == "any"


def test_load_from_file(tmp_path: Path) -> None:
    path = tmp_path / "g.toml"
    path.write_text(
        '[settings]\nstable_frames = 3\ncooldown = 1\nhand = "Right"\n'
        '[actions]\nfist = "toggle"\nrock_on = "key:ctrl+shift+t"\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.stable_frames == 3
    assert config.cooldown == 1.0
    assert config.hand == "Right"
    assert config.actions["rock_on"].keys == ("ctrl", "shift", "t")


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="init-config"):
        load_config(tmp_path / "nope.toml")


def test_without_toggle_actions_are_always_armed() -> None:
    config = parse_config('[settings]\nstart_armed = false\n[actions]\nfist = "key:space"\n')
    assert config.toggle_gesture is None
    assert config.start_armed is True


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[actions\n", "invalid TOML"),
        ("[actions]\n", "no actions"),
        ('[extra]\n[actions]\nfist = "toggle"\n', "unknown section"),
        ('[actions]\n"Bad Name" = "toggle"\n', "invalid gesture name"),
        ('[actions]\nnone = "toggle"\n', "invalid gesture name"),
        ("[actions]\nfist = 3\n", "must be a string"),
        ('[actions]\nfist = "fly"\n', "fist: unknown action"),
        ('[actions]\nfist = "toggle"\nok = "toggle"\n', "only one toggle"),
        ('[settings]\nspeed = 1\n[actions]\nfist = "toggle"\n', "unknown setting"),
        ('[settings]\nstart_armed = "yes"\n[actions]\nfist = "toggle"\n', "start_armed"),
        ('[settings]\nstable_frames = 0\n[actions]\nfist = "toggle"\n', "stable_frames"),
        ('[settings]\nstable_frames = true\n[actions]\nfist = "toggle"\n', "stable_frames"),
        ('[settings]\ncooldown = -1\n[actions]\nfist = "toggle"\n', "cooldown"),
        ('[settings]\nhand = "both"\n[actions]\nfist = "toggle"\n', "hand must be"),
        ('settings = 1\n[actions]\nfist = "toggle"\n', "must be tables"),
    ],
)
def test_invalid_configs(text: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(text)
