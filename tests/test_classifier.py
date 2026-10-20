from pathlib import Path

import numpy as np
import pytest

from gesture_pilot.classifier import (
    ALGORITHMS,
    ClassifierError,
    GestureClassifier,
    build_pipeline,
    format_confusion,
)
from gesture_pilot.dataset import Dataset, sample_row, synthetic_samples
from gesture_pilot.gestures import NONE
from gesture_pilot.synthetic import POSES, make_pose


def _dataset(rows: list[list[str]]) -> Dataset:
    features = np.asarray([[float(v) for v in row[2:]] for row in rows], dtype=np.float32)
    return Dataset(features, np.asarray([row[0] for row in rows]))


@pytest.fixture(scope="module")
def synthetic() -> Dataset:
    return _dataset(synthetic_samples(30, seed=3))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_train_learns_the_builtin_poses(synthetic: Dataset, algorithm: str) -> None:
    classifier, report = GestureClassifier.train(synthetic, algorithm=algorithm)
    assert report.labels == tuple(sorted(POSES))
    assert report.samples == len(synthetic) == report.train_samples + report.test_samples
    assert report.accuracy >= 0.9
    assert report.confusion.shape == (len(POSES), len(POSES))
    assert "precision" in report.report

    rng = np.random.default_rng(42)
    for name in POSES:
        hand = make_pose(name, handedness="Left", center=(0.4, 0.5), scale=0.2, jitter=0.02, rng=rng)
        prediction = classifier.predict(hand)
        assert prediction.gesture == name
        assert 0.0 < prediction.confidence <= 1.0


def test_low_confidence_predicts_none(synthetic: Dataset) -> None:
    classifier, _ = GestureClassifier.train(synthetic, algorithm="knn")
    classifier.min_confidence = 1.01  # nothing can be that sure
    assert classifier.predict(make_pose("fist")).gesture == NONE


def test_save_and_load_round_trip(synthetic: Dataset, tmp_path: Path) -> None:
    classifier, _ = GestureClassifier.train(synthetic, algorithm="knn")
    path = tmp_path / "models" / "g.joblib"
    classifier.save(path)

    loaded = GestureClassifier.load(path, min_confidence=0.3)
    assert loaded.labels == classifier.labels
    assert loaded.algorithm == "knn"
    assert loaded.min_confidence == 0.3
    hand = make_pose("thumbs_up")
    assert loaded.predict(hand) == classifier.predict(hand)


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(ClassifierError, match="not found"):
        GestureClassifier.load(tmp_path / "missing.joblib")

    junk = tmp_path / "junk.joblib"
    junk.write_bytes(b"not a pickle")
    with pytest.raises(ClassifierError, match="cannot load"):
        GestureClassifier.load(junk)

    import joblib

    foreign = tmp_path / "foreign.joblib"
    joblib.dump({"hello": "world"}, foreign)
    with pytest.raises(ClassifierError, match="not a gesture-pilot classifier"):
        GestureClassifier.load(foreign)


def test_train_validates_the_dataset(synthetic: Dataset) -> None:
    one_class = _dataset([sample_row("fist", make_pose("fist")) for _ in range(20)])
    with pytest.raises(ClassifierError, match="at least two"):
        GestureClassifier.train(one_class)

    rows = [sample_row("fist", make_pose("fist")) for _ in range(20)]
    rows += [sample_row("peace", make_pose("peace")) for _ in range(3)]
    with pytest.raises(ClassifierError, match="too few for: peace"):
        GestureClassifier.train(_dataset(rows))

    with pytest.raises(ClassifierError, match="test_size"):
        GestureClassifier.train(synthetic, test_size=0.9)
    with pytest.raises(ClassifierError, match="unknown algorithm"):
        build_pipeline("deep-magic")


def test_predict_rejects_wrong_feature_size(synthetic: Dataset) -> None:
    classifier, _ = GestureClassifier.train(synthetic, algorithm="knn")
    with pytest.raises(ClassifierError, match="expected 63"):
        classifier.predict_features(np.zeros(10))


def test_format_confusion_lists_every_label(synthetic: Dataset) -> None:
    _, report = GestureClassifier.train(synthetic, algorithm="knn")
    text = format_confusion(report)
    assert len(text.splitlines()) == len(POSES) + 1
    for label in POSES:
        assert label in text
