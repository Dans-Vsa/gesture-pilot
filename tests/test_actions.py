import pytest

from gesture_pilot.actions import (
    SPECIAL_KEYS,
    Action,
    ActionError,
    DryRunBackend,
    parse_action,
    perform,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("key:space", Action("key", keys=("space",))),
        ("key:ctrl+shift+t", Action("key", keys=("ctrl", "shift", "t"))),
        ("KEY:Control+C", Action("key", keys=("ctrl", "c"))),
        ("key:volume_up", Action("key", keys=("media_volume_up",))),
        ("key:media_play_pause", Action("key", keys=("media_play_pause",))),
        ("key:f5", Action("key", keys=("f5",))),
        ("click", Action("click")),
        ("click:right", Action("click", button="right")),
        ("click:left:2", Action("click", count=2)),
        ("scroll:-3", Action("scroll", dy=-3)),
        ("scroll:2, 1", Action("scroll", dx=2, dy=1)),
        ("move", Action("move")),
        ("move:900", Action("move", speed=900.0)),
        ("toggle", Action("toggle")),
    ],
)
def test_parse_action(spec: str, expected: Action) -> None:
    action = parse_action(spec)
    assert action == expected
    assert str(action) == spec.strip()


@pytest.mark.parametrize(
    "spec",
    ["", "jump", "key", "key:", "key:hyper", "key:ctrl+", "click:side", "click:left:9",
     "click:left:x", "scroll", "scroll:a", "scroll:1,2,3", "move:fast", "move:-1", "toggle:now"],
)
def test_parse_action_rejects_invalid(spec: str) -> None:
    with pytest.raises(ActionError):
        parse_action(spec)


def test_only_move_is_continuous() -> None:
    assert parse_action("move").continuous
    assert not parse_action("key:a").continuous
    assert not parse_action("toggle").continuous


def test_perform_with_dry_run_backend() -> None:
    backend = DryRunBackend()
    for spec in ("key:ctrl+c", "click:right", "click:left:2", "scroll:0,-3"):
        perform(parse_action(spec), backend)
    assert backend.log == ["press ctrl+c", "click right", "click left x2", "scroll 0,-3"]

    with pytest.raises(ActionError):
        perform(parse_action("move"), backend)


def test_special_keys_exist_in_pynput() -> None:
    try:
        from pynput.keyboard import Key
    except Exception:  # no display (e.g. headless Linux CI)
        pytest.skip("pynput is not usable here")
    missing = sorted(name for name in SPECIAL_KEYS if not hasattr(Key, name))
    assert not missing
