from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gesture_pilot.model import MODEL_FILENAME, verify_model
from gesture_pilot.overlay import draw_hand
from gesture_pilot.synthetic import make_pose
from gesture_pilot.tracker import HandTracker, TrackerError, hands_from_result

LOCAL_MODEL = Path(__file__).resolve().parents[1] / "models" / MODEL_FILENAME
needs_model = pytest.mark.skipif(not verify_model(LOCAL_MODEL), reason="hand landmarker model not downloaded")


def _fake_result(n_hands: int, labels: list[str]) -> SimpleNamespace:
    marks = [[SimpleNamespace(x=0.1 * h + i / 100, y=0.5, z=0.0) for i in range(21)] for h in range(n_hands)]
    handed = [[SimpleNamespace(category_name=label, score=0.9)] for label in labels]
    return SimpleNamespace(hand_landmarks=marks, handedness=handed)


def test_hands_from_result_converts_landmarks_and_handedness() -> None:
    hands = hands_from_result(_fake_result(2, ["Left", "Right"]))
    assert [h.handedness for h in hands] == ["Left", "Right"]
    assert hands[0].points.shape == (21, 3)
    assert hands[1].points[0, 0] == pytest.approx(0.1)
    assert hands[0].score == pytest.approx(0.9)


def test_hands_from_result_handles_empty_and_missing_handedness() -> None:
    assert hands_from_result(SimpleNamespace(hand_landmarks=[], handedness=[])) == []
    hands = hands_from_result(_fake_result(1, []))
    assert hands[0].handedness == "Right"


def test_tracker_reports_missing_model(tmp_path: Path) -> None:
    with pytest.raises(TrackerError, match="download-model"):
        HandTracker(tmp_path / "missing.task")


@needs_model
def test_tracker_runs_on_blank_frames_with_repeated_timestamps() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    with HandTracker(LOCAL_MODEL, max_hands=1) as tracker:
        assert tracker.detect(frame, 0) == []
        assert tracker.detect(frame, 0) == []  # timestamp is bumped, not rejected


def test_draw_hand_draws_skeleton_and_label() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hand(frame, make_pose("open_palm", center=(0.5, 0.7), scale=0.25), label="Right: open_palm")
    assert frame.any()
    assert frame[:, :, 1].sum() > 0
    # Nothing is drawn far away from the hand.
    assert not frame[:, :40].any()


def test_missing_system_library_becomes_tracker_error(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from mediapipe.tasks.python import vision

    model_file = tmp_path / "hand.task"
    model_file.write_bytes(b"not really a model")

    def boom(options: object) -> None:
        raise OSError("libEGL.so.1: cannot open shared object file")

    monkeypatch.setattr(vision.HandLandmarker, "create_from_options", boom)
    with pytest.raises(TrackerError, match="libegl1"):
        HandTracker(model_file)
