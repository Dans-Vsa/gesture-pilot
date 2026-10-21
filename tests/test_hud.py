"""On-screen HUD text and drawing."""

import numpy as np

from gesture_pilot.hud import ACTION_HIGHLIGHT, AMBER, GREEN, GREY, HudState, draw_hud


def texts(state: HudState, now: float = 0.0) -> list[str]:
    return [text for text, _ in state.lines(now)]


def test_minimal_hud_shows_mode_only() -> None:
    assert texts(HudState("track")) == ["mode: track"]


def test_hud_shows_performance_options_and_dry_run() -> None:
    state = HudState("run", process_every=2, detect_size=(320, 240), dry_run=True)
    assert texts(state)[0] == "mode: run (dry run)  [detect 1/2, 320x240]"


def test_hud_armed_state_colours() -> None:
    armed = HudState("run", armed=True).lines(0)
    disarmed = HudState("run", armed=False).lines(0)
    assert armed[1] == ("ARMED", GREEN)
    assert disarmed[1] == ("disarmed", GREY)


def test_hud_lists_gestures_sorted_by_hand() -> None:
    state = HudState("recognize", gestures={"Right": "fist", "Left": "peace"})
    assert texts(state)[1] == "Left: peace  Right: fist"


def test_last_action_is_highlighted_then_fades() -> None:
    state = HudState("run", armed=True)
    state.set_action("open_palm -> key:media_play_pause", now=10.0)
    assert state.lines(10.5)[-1] == ("last: open_palm -> key:media_play_pause", AMBER)
    assert state.lines(10.0 + ACTION_HIGHLIGHT + 0.1)[-1][1] == GREY


def test_draw_hud_darkens_bottom_left_and_writes_text() -> None:
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    state = HudState("run", armed=True, gestures={"Right": "point"})
    state.set_action("armed", now=0.0)
    draw_hud(frame, state, now=0.0)
    assert frame.shape == (240, 320, 3)
    assert (frame[0:100, :] == 200).all()  # top untouched
    corner = frame[-6:, :6]
    assert int(corner.max()) < 200  # panel background darkened
    assert len(np.unique(frame[150:, :200].reshape(-1, 3), axis=0)) > 2  # text drawn


def test_draw_hud_fits_tiny_frames() -> None:
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    state = HudState("run", armed=False, gestures={"Left": "thumbs_down", "Right": "thumbs_up"})
    draw_hud(frame, state, now=0.0)
    assert frame.shape == (40, 60, 3)
