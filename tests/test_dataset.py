from pathlib import Path

import numpy as np
import pytest

from gesture_pilot.dataset import (
    FEATURE_SIZE,
    HEADER,
    DatasetError,
    append_samples,
    load_dataset,
    sample_row,
    synthetic_samples,
    validate_label,
)
from gesture_pilot.landmarks import feature_vector
from gesture_pilot.synthetic import POSES, make_pose


def test_sample_row_holds_label_handedness_and_features() -> None:
    hand = make_pose("peace", handedness="Left")
    row = sample_row("peace", hand)
    assert row[:2] == ["peace", "Left"]
    assert len(row) == len(HEADER) == FEATURE_SIZE + 2
    np.testing.assert_allclose([float(v) for v in row[2:]], feature_vector(hand), atol=1e-4)


def test_append_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "data.csv"
    assert append_samples(path, [sample_row("fist", make_pose("fist"))]) == 1
    assert append_samples(path, [sample_row("point", make_pose("point")) for _ in range(3)]) == 3
    assert path.read_text(encoding="utf-8").count("label,handedness") == 1  # header written once

    data = load_dataset(path)
    assert len(data) == 4
    assert data.features.shape == (4, FEATURE_SIZE)
    assert data.features.dtype == np.float32
    assert data.counts() == {"fist": 1, "point": 3}


def test_append_refuses_foreign_csv(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="unexpected header"):
        append_samples(path, [sample_row("fist", make_pose("fist"))])


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.csv")

    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(HEADER) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no samples"):
        load_dataset(empty)

    short = tmp_path / "short.csv"
    short.write_text(",".join(HEADER) + "\nfist,Right,1,2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="short.csv:2: expected"):
        load_dataset(short)

    bad = tmp_path / "bad.csv"
    row = sample_row("fist", make_pose("fist"))
    row[5] = "oops"
    bad.write_text(",".join(HEADER) + "\n" + ",".join(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="non-numeric"):
        load_dataset(bad)


@pytest.mark.parametrize("label", ["rock_on", "swipe-left", "a1"])
def test_valid_labels(label: str) -> None:
    assert validate_label(label) == label


@pytest.mark.parametrize("label", ["", "Rock", "two words", "none", "_x", "ok!"])
def test_invalid_labels(label: str) -> None:
    with pytest.raises(DatasetError):
        validate_label(label)


def test_synthetic_samples_cover_all_poses_and_are_reproducible() -> None:
    rows = synthetic_samples(5, seed=1)
    assert len(rows) == 5 * len(POSES)
    assert {row[0] for row in rows} == set(POSES)
    assert {row[1] for row in rows} <= {"Left", "Right"}
    assert rows == synthetic_samples(5, seed=1)
    assert rows != synthetic_samples(5, seed=2)
    assert len(synthetic_samples(3, poses=["fist"])) == 3
