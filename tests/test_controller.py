import pytest

from gesture_pilot.actions import DryRunBackend
from gesture_pilot.config import parse_config
from gesture_pilot.controller import GestureController
from gesture_pilot.landmarks import INDEX_TIP
from gesture_pilot.smoothing import GestureEvent
from gesture_pilot.synthetic import make_pose

CONFIG = """
[settings]
hand = "any"
[actions]
ok = "toggle"
open_palm = "key:media_play_pause"
pinch = "click:left"
point = "move:1000"
"""


def _controller(text: str = CONFIG) -> tuple[GestureController, DryRunBackend]:
    backend = DryRunBackend()
    return GestureController(parse_config(text), backend), backend


def _event(gesture: str, hand: str = "Right") -> GestureEvent:
    return GestureEvent(gesture, hand, 0.0)


def test_starts_disarmed_and_ignores_actions() -> None:
    controller, backend = _controller()
    assert not controller.armed
    assert controller.handle_event(_event("open_palm")) == "open_palm -> key:media_play_pause (ignored: disarmed)"
    assert backend.log == []


def test_toggle_arms_and_disarms() -> None:
    controller, backend = _controller()
    assert controller.handle_event(_event("ok")) == "armed"
    assert controller.handle_event(_event("open_palm")) == "open_palm -> key:media_play_pause"
    assert controller.handle_event(_event("pinch")) == "pinch -> click:left"
    assert controller.handle_event(_event("ok")) == "disarmed"
    controller.handle_event(_event("open_palm"))
    assert backend.log == ["press media_play_pause", "click left"]


def test_unmapped_and_continuous_gestures_give_no_event_action() -> None:
    controller, backend = _controller()
    controller.armed = True
    assert controller.handle_event(_event("fist")) is None
    assert controller.handle_event(_event("point")) is None
    assert backend.log == []


def test_hand_filter() -> None:
    controller, backend = _controller(CONFIG.replace('hand = "any"', 'hand = "Left"'))
    assert controller.handle_event(_event("ok", "Right")) is None
    assert controller.handle_event(_event("ok", "Left")) == "armed"
    controller.handle_event(_event("open_palm", "Right"))
    controller.handle_event(_event("open_palm", "Left"))
    assert backend.log == ["press media_play_pause"]


def _point(x: float, y: float, hand: str = "Right"):
    pose = make_pose("point", handedness=hand)
    shift = (x, y) - pose.points[INDEX_TIP, :2]
    pose.points[:, :2] += shift
    return pose


def test_move_follows_the_index_tip_while_held() -> None:
    controller, backend = _controller()
    controller.armed = True
    labels = {"Right": "point"}
    controller.update_frame([_point(0.50, 0.50)], labels)  # first frame only anchors
    controller.update_frame([_point(0.52, 0.49)], labels)
    controller.update_frame([_point(0.5205, 0.4901)], labels)  # below the dead zone
    controller.update_frame([_point(0.53, 0.49)], labels)
    assert backend.log == ["move 20,-10", "move 10,0"]


def test_move_resets_when_the_gesture_changes() -> None:
    controller, backend = _controller()
    controller.armed = True
    controller.update_frame([_point(0.5, 0.5)], {"Right": "point"})
    controller.update_frame([_point(0.6, 0.5)], {"Right": "fist"})  # released: no move, anchor dropped
    controller.update_frame([_point(0.7, 0.5)], {"Right": "point"})  # re-anchors, no jump
    controller.update_frame([_point(0.71, 0.5)], {"Right": "point"})
    assert backend.log == ["move 10,0"]


@pytest.mark.parametrize("armed", [False, True])
def test_move_needs_arming_and_an_allowed_hand(armed: bool) -> None:
    controller, backend = _controller(CONFIG.replace('hand = "any"', 'hand = "Left"'))
    controller.armed = armed
    for x in (0.5, 0.6):
        controller.update_frame([_point(x, 0.5, "Right")], {"Right": "point"})
    assert backend.log == []
