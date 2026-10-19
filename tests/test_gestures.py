import numpy as np
import pytest

from gesture_pilot.gestures import GESTURES, NONE, analyze, classify, classify_pose
from gesture_pilot.synthetic import POSES, make_hand, make_pose


@pytest.mark.parametrize("handedness", ["Right", "Left"])
@pytest.mark.parametrize("name", GESTURES)
def test_every_builtin_gesture_is_recognised(name: str, handedness: str) -> None:
    result = classify(make_pose(name, handedness=handedness))
    assert result.gesture == name
    assert result.handedness == handedness


def test_poses_and_gestures_match() -> None:
    assert set(POSES) == set(GESTURES)


@pytest.mark.parametrize("name", GESTURES)
def test_recognition_is_robust_to_noise_rotation_and_scale(name: str) -> None:
    rng = np.random.default_rng(42)
    hits = 0
    trials = 100
    for _ in range(trials):
        hand = make_pose(
            name,
            handedness=str(rng.choice(["Left", "Right"])),
            rotation=float(rng.uniform(-15, 15)),
            scale=float(rng.uniform(0.08, 0.3)),
            center=(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.4, 0.7))),
            jitter=0.03,
            rng=rng,
        )
        hits += classify(hand).gesture == name
    assert hits / trials >= 0.95


def test_finger_states_of_open_palm_and_fist() -> None:
    assert analyze(make_pose("open_palm")).extended == ("thumb", "index", "middle", "ring", "pinky")
    assert analyze(make_pose("fist")).extended == ()


def test_pinch_distance_is_small_only_when_pinching() -> None:
    assert analyze(make_pose("pinch")).pinch_distance < 0.25
    assert analyze(make_pose("open_palm")).pinch_distance > 0.5


def test_sideways_thumb_is_not_thumbs_up_or_down() -> None:
    hand = make_hand({"thumb": True}, rotation=-45)  # thumb points to the side
    assert classify(hand).gesture == NONE


def test_unlisted_finger_combination_is_none() -> None:
    hand = make_hand({"index": True, "pinky": True})  # "rock on"
    assert classify(hand).gesture == NONE


def test_upside_down_open_palm_is_still_open_palm() -> None:
    assert classify(make_pose("open_palm", rotation=180)).gesture == "open_palm"


def test_classify_pose_prefers_ok_over_open_palm_when_tips_touch() -> None:
    pose = analyze(make_pose("ok"))
    assert pose.fingers["middle"] and pose.fingers["ring"] and pose.fingers["pinky"]
    assert classify_pose(pose) == "ok"
