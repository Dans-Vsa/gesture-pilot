"""Rule-based gesture recognition from hand landmarks.

The rules work on :func:`gesture_pilot.landmarks.normalize`d landmarks, so they
do not depend on where the hand is in the frame, how big it is, or whether it
is a left or right hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gesture_pilot.landmarks import (
    FINGERS,
    INDEX_TIP,
    PINKY_MCP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    Hand,
    normalize,
)

NONE = "none"

GESTURES = (
    "open_palm",
    "fist",
    "point",
    "peace",
    "thumbs_up",
    "thumbs_down",
    "ok",
    "pinch",
)

#: A finger is "straight" when its bend angle (degrees) is below this.
STRAIGHT_ANGLE = 55.0
#: Thumb and index tips closer than this (in palm sizes) count as touching.
PINCH_DISTANCE = 0.25
#: Minimum vertical component of the thumb direction for thumbs up/down.
THUMB_VERTICAL = 0.7


@dataclass(frozen=True)
class HandPose:
    """Geometric summary of a hand used by the rules."""

    fingers: dict[str, bool]  # finger name -> extended
    pinch_distance: float  # thumb tip ↔ index tip, in palm sizes
    thumb_direction: np.ndarray  # unit vector thumb MCP -> tip (y up)

    @property
    def extended(self) -> tuple[str, ...]:
        return tuple(name for name, up in self.fingers.items() if up)


@dataclass(frozen=True)
class GestureResult:
    gesture: str
    pose: HandPose
    handedness: str


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def finger_states(pts: np.ndarray) -> dict[str, bool]:
    """Which fingers are extended, from normalised ``(21, 3)`` landmarks."""
    states: dict[str, bool] = {}
    for name, (base, mid, dip, tip) in FINGERS.items():
        if name == "thumb":
            bend = _angle(pts[mid] - pts[base], pts[tip] - pts[dip])
            reach = np.linalg.norm(pts[tip] - pts[PINKY_MCP]) > np.linalg.norm(pts[dip] - pts[PINKY_MCP])
            states[name] = bend < STRAIGHT_ANGLE and bool(reach)
        else:
            bend = _angle(pts[mid] - pts[base], pts[tip] - pts[mid])
            reach = np.linalg.norm(pts[tip] - pts[WRIST]) > np.linalg.norm(pts[mid] - pts[WRIST])
            states[name] = bend < STRAIGHT_ANGLE and bool(reach)
    return states


def analyze(hand: Hand) -> HandPose:
    pts = normalize(hand)
    direction = pts[THUMB_TIP, :2] - pts[THUMB_MCP, :2]
    norm = float(np.linalg.norm(direction))
    unit = direction / norm if norm > 1e-9 else np.zeros(2, dtype=np.float32)
    return HandPose(
        fingers=finger_states(pts),
        pinch_distance=float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])),
        thumb_direction=unit,
    )


def classify_pose(pose: HandPose) -> str:
    """Map a :class:`HandPose` to a gesture name (or :data:`NONE`)."""
    f = pose.fingers
    others = (f["middle"], f["ring"], f["pinky"])

    if pose.pinch_distance < PINCH_DISTANCE:
        if all(others):
            return "ok"
        if not any(others):
            return "pinch"

    four = (f["index"], *others)
    if f["thumb"] and all(four):
        return "open_palm"
    if f["thumb"] and not any(four):
        dy = float(pose.thumb_direction[1])
        if dy > THUMB_VERTICAL:
            return "thumbs_up"
        if dy < -THUMB_VERTICAL:
            return "thumbs_down"
        return NONE
    if not f["thumb"] and not any(four):
        return "fist"
    if f["index"] and not any(others):
        return "point"
    if f["index"] and f["middle"] and not f["ring"] and not f["pinky"]:
        return "peace"
    return NONE


def classify(hand: Hand) -> GestureResult:
    """Recognise the gesture of one hand."""
    pose = analyze(hand)
    return GestureResult(classify_pose(pose), pose, hand.handedness)
