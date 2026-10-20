"""Trainable gesture classifier (scikit-learn) for custom gestures.

The classifier sees the same 63 normalised landmark features as the rules, so
a few hundred samples per gesture recorded with ``gesture-pilot record`` are
enough. Models are saved with joblib together with their label list and the
gesture-pilot version.

.. warning:: joblib files are pickles — only load models you trained yourself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_pilot import __version__
from gesture_pilot.dataset import FEATURE_SIZE, Dataset
from gesture_pilot.gestures import NONE
from gesture_pilot.landmarks import Hand, feature_vector

ALGORITHMS = ("svm", "knn", "forest")
FORMAT = "gesture-pilot-classifier/1"
MIN_SAMPLES = 10  # per gesture


class ClassifierError(Exception):
    """Training failed or a model file is invalid."""


def build_pipeline(algorithm: str = "svm", *, seed: int = 0):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if algorithm == "svm":
        # Calibrated probabilities (SVC(probability=True) is deprecated since scikit-learn 1.9).
        model = CalibratedClassifierCV(SVC(kernel="rbf", C=10.0, gamma="scale"), cv=3, ensemble=False)
    elif algorithm == "knn":
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")
    elif algorithm == "forest":
        model = RandomForestClassifier(n_estimators=200, random_state=seed)
    else:
        raise ClassifierError(f"unknown algorithm {algorithm!r}; choose from {', '.join(ALGORITHMS)}")
    return make_pipeline(StandardScaler(), model)


@dataclass(frozen=True)
class TrainReport:
    algorithm: str
    samples: int
    train_samples: int
    test_samples: int
    accuracy: float
    labels: tuple[str, ...]
    report: str  # sklearn classification report (text)
    confusion: np.ndarray


@dataclass(frozen=True)
class Prediction:
    gesture: str
    confidence: float


class GestureClassifier:
    """A fitted pipeline plus its labels; predicts ``none`` below ``min_confidence``."""

    def __init__(self, pipeline, labels: tuple[str, ...], algorithm: str, min_confidence: float = 0.6) -> None:
        self.pipeline = pipeline
        self.labels = labels
        self.algorithm = algorithm
        self.min_confidence = min_confidence

    @classmethod
    def train(
        cls,
        dataset: Dataset,
        *,
        algorithm: str = "svm",
        test_size: float = 0.2,
        seed: int = 0,
    ) -> tuple[GestureClassifier, TrainReport]:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.model_selection import train_test_split

        counts = dataset.counts()
        if len(counts) < 2:
            raise ClassifierError("need samples of at least two different gestures to train")
        too_few = [label for label, n in counts.items() if n < MIN_SAMPLES]
        if too_few:
            raise ClassifierError(
                f"need at least {MIN_SAMPLES} samples per gesture; too few for: {', '.join(too_few)}"
            )
        if not 0 < test_size <= 0.5:
            raise ClassifierError("test_size must be in (0, 0.5]")

        x_train, x_test, y_train, y_test = train_test_split(
            dataset.features, dataset.labels, test_size=test_size, random_state=seed, stratify=dataset.labels
        )
        pipeline = build_pipeline(algorithm, seed=seed)
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        labels = tuple(str(c) for c in pipeline.classes_)
        report = TrainReport(
            algorithm=algorithm,
            samples=len(dataset),
            train_samples=len(y_train),
            test_samples=len(y_test),
            accuracy=float(accuracy_score(y_test, predicted)),
            labels=labels,
            report=classification_report(y_test, predicted, labels=list(labels), zero_division=0),
            confusion=confusion_matrix(y_test, predicted, labels=list(labels)),
        )
        # Refit on all samples so the saved model uses every recording.
        pipeline.fit(dataset.features, dataset.labels)
        return cls(pipeline, labels, algorithm), report

    def predict_features(self, features: np.ndarray) -> Prediction:
        features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if features.shape[1] != FEATURE_SIZE:
            raise ClassifierError(f"expected {FEATURE_SIZE} features, got {features.shape[1]}")
        probabilities = self.pipeline.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        if confidence < self.min_confidence:
            return Prediction(NONE, confidence)
        return Prediction(str(self.pipeline.classes_[best]), confidence)

    def predict(self, hand: Hand) -> Prediction:
        return self.predict_features(feature_vector(hand))

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": FORMAT,
                "version": __version__,
                "algorithm": self.algorithm,
                "labels": list(self.labels),
                "pipeline": self.pipeline,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, min_confidence: float = 0.6) -> GestureClassifier:
        import joblib

        if not path.is_file():
            raise ClassifierError(f"classifier not found: {path} (train one with `gesture-pilot train`)")
        try:
            data = joblib.load(path)
        except Exception as exc:  # corrupt / foreign file
            raise ClassifierError(f"cannot load classifier {path}: {exc}") from None
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ClassifierError(f"{path} is not a gesture-pilot classifier")
        return cls(data["pipeline"], tuple(data["labels"]), data["algorithm"], min_confidence)


def format_confusion(report: TrainReport) -> str:
    """Confusion matrix as a small text table (rows = true, columns = predicted)."""
    width = max(6, *(len(label) for label in report.labels))
    head = " " * width + " " + " ".join(f"{i:>4}" for i in range(len(report.labels)))
    lines = [head]
    for i, (label, row) in enumerate(zip(report.labels, report.confusion)):
        cells = " ".join(f"{int(v):>4}" for v in row)
        lines.append(f"{label:>{width}} {cells}   [{i}]")
    return "\n".join(lines)
