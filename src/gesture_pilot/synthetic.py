"""Generate synthetic hand landmarks for tests and demo data.

The generator builds a plausible 21-point hand from a set of finger states
(extended / curled, optional thumb-index pinch), then rotates, scales,
translates and jitters it. It is not anatomically perfect, but it produces the
same geometric cues a real MediaPipe hand does, which is all the gesture rules
and classifiers look at.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from gesture_pilot.landmarks import FINGER_NAMES, Hand

# Local hand frame: wrist at the origin, x to the thumb side, y up, palm size 1.
_BASES = {
    "thumb": np.array([0.25, 0.25, 0.0]),  # CMC
    "index": np.array([0.30, 0.95, 0.0]),
    "middle": np.array([0.08, 1.00, 0.0]),
    "ring": np.array([-0.12, 0.95, 0.0]),
    "pinky": np.array([-0.30, 0.85, 0.0]),
}
_SEGMENTS = {"index": (0.40, 0.25, 0.20), "middle": (0.45, 0.28, 0.22),
             "ring": (0.40, 0.26, 0.20), "pinky": (0.30, 0.20, 0.18)}
_SPREAD = {"index": 0.12, "middle": 0.0, "ring": -0.10, "pinky": -0.22}  # radians

_PINCH_POINT = np.array([0.62, 0.86, -0.05])

#: Finger states of the built-in rule-based gestures, plus hand rotation (deg).
POSES: dict[str, dict[str, object]] = {
    "open_palm": {"fingers": dict.fromkeys(FINGER_NAMES, True)},
    "fist": {"fingers": dict.fromkeys(FINGER_NAMES, False)},
    "point": {"fingers": {"index": True}},
    "peace": {"fingers": {"index": True, "middle": True}},
    "thumbs_up": {"fingers": {"thumb": True}, "rotation": 45.0},
    "thumbs_down": {"fingers": {"thumb": True}, "rotation": 225.0},
    "ok": {"fingers": {"middle": True, "ring": True, "pinky": True}, "pinch": True},
    "pinch": {"fingers": {}, "pinch": True},
}


def _finger(name: str, extended: bool) -> list[np.ndarray]:
    base = _BASES[name]
    if name == "thumb":
        mcp = np.array([0.45, 0.45, 0.0])
        if extended:
            direction = np.array([math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0])
            ip = mcp + 0.30 * direction
            tip = ip + 0.25 * direction
        else:  # folded across the palm
            ip = mcp + np.array([0.05, 0.25, -0.05])
            tip = np.array([0.10, 0.60, -0.10])
        return [base, mcp, ip, tip]

    a, b, c = _SEGMENTS[name]
    angle = _SPREAD[name]
    up = np.array([-math.sin(angle), math.cos(angle), 0.0])
    if extended:
        pip = base + a * up
        dip = pip + b * up
        tip = dip + c * up
    else:  # curled towards the palm (and the camera)
        pip = base + a * 0.85 * up
        dip = pip + np.array([0.0, -b * 0.6, -b * 0.8])
        tip = dip + np.array([0.0, -c * 0.9, -c * 0.4])
    return [base, pip, dip, tip]


def _pinch_fingers() -> tuple[list[np.ndarray], list[np.ndarray]]:
    thumb = [_BASES["thumb"], np.array([0.45, 0.45, 0.0]), np.array([0.60, 0.66, -0.02]),
             _PINCH_POINT + np.array([0.0, -0.02, 0.0])]
    base = _BASES["index"]
    index = [base, base + np.array([0.12, 0.30, -0.02]), np.array([0.62, 1.10, -0.05]),
             _PINCH_POINT + np.array([0.01, 0.03, 0.0])]
    return thumb, index


def make_hand(
    fingers: Mapping[str, bool] | None = None,
    *,
    pinch: bool = False,
    rotation: float = 0.0,
    handedness: str = "Right",
    center: tuple[float, float] = (0.5, 0.6),
    scale: float = 0.18,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Hand:
    """Build a synthetic :class:`Hand`.

    ``fingers`` maps finger name → extended (missing fingers are curled).
    ``rotation`` turns the hand counter-clockwise in degrees (0 = fingers up).
    ``scale`` is the palm size as a fraction of the image height; ``jitter``
    adds Gaussian noise in palm-size units.
    """
    states = {name: bool((fingers or {}).get(name, False)) for name in FINGER_NAMES}
    joints: dict[str, list[np.ndarray]] = {n: _finger(n, states[n]) for n in FINGER_NAMES}
    if pinch:
        joints["thumb"], joints["index"] = _pinch_fingers()

    local = np.zeros((21, 3), dtype=np.float64)
    for i, name in enumerate(FINGER_NAMES):
        local[1 + 4 * i: 5 + 4 * i] = joints[name]

    if jitter > 0:
        rng = rng or np.random.default_rng()
        local = local + rng.normal(0.0, jitter, local.shape)
        local[0] = 0.0

    theta = math.radians(rotation)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    xy = local[:, :2] @ rot.T
    if handedness == "Left":
        xy[:, 0] *= -1.0

    points = np.empty((21, 3), dtype=np.float32)
    points[:, 0] = center[0] + xy[:, 0] * scale
    points[:, 1] = center[1] - xy[:, 1] * scale  # image y points down
    points[:, 2] = local[:, 2] * scale
    return Hand(points, handedness=handedness)


def make_pose(name: str, **kwargs: object) -> Hand:
    """Synthetic hand for one of the built-in gestures in :data:`POSES`."""
    try:
        pose = POSES[name]
    except KeyError:
        raise ValueError(f"unknown pose {name!r}; choose from {', '.join(POSES)}") from None
    params: dict[str, object] = {"rotation": pose.get("rotation", 0.0), "pinch": pose.get("pinch", False)}
    params.update(kwargs)
    if "rotation" in kwargs:
        params["rotation"] = float(pose.get("rotation", 0.0)) + float(kwargs["rotation"])  # type: ignore[arg-type]
    return make_hand(pose["fingers"], **params)  # type: ignore[arg-type]
