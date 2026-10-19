import numpy as np
import pytest

from gesture_pilot.gestures import NONE
from gesture_pilot.landmarks import Hand
from gesture_pilot.smoothing import GestureStabilizer, LandmarkSmoother, MultiHandStabilizer


def _hand(value: float, handedness: str = "Right") -> Hand:
    return Hand(np.full((21, 3), value, dtype=np.float32), handedness=handedness)


def test_smoother_moving_average() -> None:
    smoother = LandmarkSmoother(alpha=0.5)
    first = smoother.update([_hand(0.0)])[0]
    second = smoother.update([_hand(1.0)])[0]
    third = smoother.update([_hand(1.0)])[0]
    assert first.points[0, 0] == pytest.approx(0.0)
    assert second.points[0, 0] == pytest.approx(0.5)
    assert third.points[0, 0] == pytest.approx(0.75)


def test_smoother_resets_when_hand_disappears() -> None:
    smoother = LandmarkSmoother(alpha=0.5)
    smoother.update([_hand(0.0)])
    smoother.update([])
    assert smoother.update([_hand(1.0)])[0].points[0, 0] == pytest.approx(1.0)


def test_smoother_tracks_hands_separately() -> None:
    smoother = LandmarkSmoother(alpha=0.5)
    smoother.update([_hand(0.0, "Left"), _hand(1.0, "Right")])
    left, right = smoother.update([_hand(1.0, "Left"), _hand(1.0, "Right")])
    assert left.points[0, 0] == pytest.approx(0.5)
    assert right.points[0, 0] == pytest.approx(1.0)


def test_smoother_alpha_one_is_passthrough_and_bad_alpha_rejected() -> None:
    smoother = LandmarkSmoother(alpha=1.0)
    smoother.update([_hand(0.0)])
    assert smoother.update([_hand(0.3)])[0].points[0, 0] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        LandmarkSmoother(alpha=0)


def _feed(stab: GestureStabilizer, labels: list[str], start: float = 0.0, dt: float = 0.1) -> list[str]:
    fired = []
    for i, label in enumerate(labels):
        event = stab.update(label, start + i * dt)
        if event:
            fired.append(event.gesture)
    return fired


def test_stabilizer_needs_n_stable_frames() -> None:
    stab = GestureStabilizer(stable_frames=3, cooldown=0)
    assert _feed(stab, ["fist", "fist"]) == []
    assert _feed(stab, ["fist"]) == ["fist"]
    assert stab.active == "fist"


def test_stabilizer_ignores_flicker() -> None:
    stab = GestureStabilizer(stable_frames=3, cooldown=0)
    assert _feed(stab, ["fist", "point", "fist", "point", "fist", "none"]) == []


def test_stabilizer_does_not_repeat_held_gesture() -> None:
    stab = GestureStabilizer(stable_frames=2, cooldown=0)
    assert _feed(stab, ["fist"] * 10) == ["fist"]


def test_stabilizer_refires_after_release() -> None:
    stab = GestureStabilizer(stable_frames=2, cooldown=0)
    assert _feed(stab, ["fist", "fist", NONE, NONE, "fist", "fist"]) == ["fist", "fist"]


def test_stabilizer_brief_dropout_does_not_refire() -> None:
    stab = GestureStabilizer(stable_frames=3, cooldown=0)
    assert _feed(stab, ["fist"] * 3 + [NONE] + ["fist"] * 3) == ["fist"]


def test_stabilizer_cooldown_blocks_quick_switch() -> None:
    stab = GestureStabilizer(stable_frames=1, cooldown=1.0)
    assert stab.update("fist", 0.0) is not None
    assert stab.update("point", 0.5) is None  # inside cooldown
    event = stab.update("point", 1.2)
    assert event is not None and event.gesture == "point" and event.timestamp == 1.2


def test_stabilizer_validates_arguments() -> None:
    with pytest.raises(ValueError):
        GestureStabilizer(stable_frames=0)
    with pytest.raises(ValueError):
        GestureStabilizer(cooldown=-1)


def test_multi_hand_stabilizer() -> None:
    multi = MultiHandStabilizer(stable_frames=2, cooldown=0)
    assert multi.update({"Left": "fist", "Right": "point"}, 0.0) == []
    events = multi.update({"Left": "fist", "Right": "point"}, 0.1)
    assert [(e.handedness, e.gesture) for e in events] == [("Left", "fist"), ("Right", "point")]
    assert multi.active("Left") == "fist"
    # Right hand leaves the frame: it is released after enough "none" frames.
    multi.update({"Left": "fist"}, 0.2)
    multi.update({"Left": "fist"}, 0.3)
    assert multi.active("Right") == NONE
    assert multi.active("Nobody") == NONE
