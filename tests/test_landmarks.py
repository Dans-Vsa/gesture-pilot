import numpy as np
import pytest

from gesture_pilot.landmarks import (
    HAND_CONNECTIONS,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    WRIST,
    Hand,
    feature_vector,
    normalize,
)
from gesture_pilot.synthetic import POSES, make_hand, make_pose


def test_hand_accepts_2d_points_and_pads_z() -> None:
    hand = Hand(np.random.default_rng(0).random((21, 2)))
    assert hand.points.shape == (21, 3)
    assert np.all(hand.points[:, 2] == 0)


@pytest.mark.parametrize("shape", [(20, 3), (21, 4), (63,)])
def test_hand_rejects_bad_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Hand(np.zeros(shape))


def test_hand_rejects_bad_handedness() -> None:
    with pytest.raises(ValueError):
        Hand(np.zeros((21, 3)), handedness="Middle")


def test_connections_cover_all_landmarks() -> None:
    used = {i for bone in HAND_CONNECTIONS for i in bone}
    assert used == set(range(NUM_LANDMARKS))
    assert len(HAND_CONNECTIONS) == 21


def test_pixels_and_bbox() -> None:
    hand = make_pose("open_palm", center=(0.5, 0.5), scale=0.2)
    px = hand.pixels(640, 480)
    assert px.dtype == np.int32 and px.shape == (21, 2)
    x0, y0, x1, y1 = hand.bbox()
    assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1
    assert tuple(px[WRIST]) == (320, 240)


def test_normalize_is_translation_and_scale_invariant() -> None:
    a = normalize(make_pose("peace", center=(0.3, 0.7), scale=0.1))
    b = normalize(make_pose("peace", center=(0.6, 0.4), scale=0.3))
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_normalize_puts_wrist_at_origin_with_unit_palm_pointing_up() -> None:
    pts = normalize(make_pose("open_palm"))
    np.testing.assert_allclose(pts[WRIST], 0, atol=1e-6)
    assert np.linalg.norm(pts[MIDDLE_MCP, :2]) == pytest.approx(1.0, abs=1e-5)
    assert pts[MIDDLE_MCP, 1] > 0.9  # fingers up => positive y after the flip


def test_normalize_mirrors_left_hands() -> None:
    right = normalize(make_pose("point", handedness="Right"))
    left = normalize(make_pose("point", handedness="Left"))
    np.testing.assert_allclose(right, left, atol=1e-5)
    raw_left = normalize(make_pose("point", handedness="Left"), mirror_left=False)
    np.testing.assert_allclose(raw_left[:, 0], -right[:, 0], atol=1e-5)


def test_normalize_rejects_degenerate_hand() -> None:
    with pytest.raises(ValueError):
        normalize(Hand(np.full((21, 3), 0.5)))


def test_feature_vector_length() -> None:
    vec = feature_vector(make_pose("fist"))
    assert vec.shape == (63,)


def test_synthetic_unknown_pose() -> None:
    with pytest.raises(ValueError, match="unknown pose"):
        make_pose("jazz_hands")


def test_synthetic_jitter_is_reproducible() -> None:
    a = make_hand({"index": True}, jitter=0.02, rng=np.random.default_rng(7))
    b = make_hand({"index": True}, jitter=0.02, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.points, b.points)
    assert set(POSES) >= {"open_palm", "fist", "ok", "pinch"}
