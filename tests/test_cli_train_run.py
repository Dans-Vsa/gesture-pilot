"""CLI tests for record / train / init-config / run (no camera, no real input)."""

from pathlib import Path

import pytest

from gesture_pilot import model
from gesture_pilot.cli import main
from gesture_pilot.config import DEFAULT_CONFIG, load_config
from gesture_pilot.dataset import load_dataset
from gesture_pilot.synthetic import POSES

LOCAL_MODEL = Path(__file__).resolve().parents[1] / "models" / model.MODEL_FILENAME
needs_model = pytest.mark.skipif(
    not model.verify_model(LOCAL_MODEL), reason="hand landmarker model not downloaded"
)


@pytest.fixture
def trained(tmp_path: Path) -> Path:
    data = tmp_path / "data.csv"
    assert main(["record", "--synthetic", "15", "--output", str(data)]) == 0
    out = tmp_path / "g.joblib"
    assert main(["train", str(data), "--algorithm", "knn", "--output", str(out)]) == 0
    return out


def test_record_synthetic_then_train(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data" / "gestures.csv"
    assert main(["record", "--synthetic", "12", "--output", str(data), "--seed", "5"]) == 0
    assert f"saved {12 * len(POSES)} synthetic samples" in capsys.readouterr().out
    assert main(["record", "--synthetic", "3", "--output", str(data)]) == 0  # appends
    assert len(load_dataset(data)) == 15 * len(POSES)

    out = tmp_path / "models" / "g.joblib"
    assert main(["train", str(data), "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "svm: trained on" in text
    assert "confusion matrix" in text
    assert "accuracy" in text and str(out) in text
    assert out.is_file()


def test_record_argument_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["record", "--source", "test", "--no-window"]) == 1
    assert "--label is required" in capsys.readouterr().err
    assert main(["record", "--label", "Bad Label", "--source", "test"]) == 1
    assert "invalid label" in capsys.readouterr().err
    assert main(["record", "--synthetic", "2", "--label", "x", "--output", str(tmp_path / "d.csv")]) == 1
    assert "do not pass --label" in capsys.readouterr().err


def test_train_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train", str(tmp_path / "missing.csv")]) == 1
    assert "dataset not found" in capsys.readouterr().err

    data = tmp_path / "small.csv"
    main(["record", "--synthetic", "4", "--output", str(data)])
    capsys.readouterr()
    assert main(["train", str(data), "--output", str(tmp_path / "m.joblib")]) == 1
    assert "at least 10 samples" in capsys.readouterr().err
    assert not (tmp_path / "m.joblib").exists()


def test_init_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "conf" / "gestures.toml"
    assert main(["init-config", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG
    assert load_config(path).toggle_gesture == "ok"
    assert main(["init-config", str(path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["init-config", str(path), "--force"]) == 0


@pytest.mark.parametrize(
    "argv",
    [["run", "--dry-run"], ["recognize"], ["record", "--label", "wave"]],
)
def test_commands_report_missing_landmarker(
    argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main([*argv, "--source", "test", "--no-window", "--model-path", str(tmp_path / "x.task")])
    assert code == 1
    assert "model not found" in capsys.readouterr().err


def test_run_and_recognize_report_bad_classifier_or_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["recognize", "--model", str(tmp_path / "none.joblib"), "--no-window"]) == 1
    assert "classifier not found" in capsys.readouterr().err

    bad = tmp_path / "bad.toml"
    bad.write_text('[actions]\nfist = "fly"\n', encoding="utf-8")
    assert main(["run", "--config", str(bad), "--dry-run", "--no-window"]) == 1
    assert "unknown action" in capsys.readouterr().err


@needs_model
def test_run_dry_run_with_trained_classifier(trained: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["run", "--dry-run", "--model", str(trained), "--source", "test", "--width", "160",
         "--height", "120", "--max-frames", "3", "--no-window", "--model-path", str(LOCAL_MODEL)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "7 gesture actions loaded (disarmed - hold 'ok' to arm), dry run" in out
    assert "0 gesture events" in out


@needs_model
def test_record_from_test_pattern_without_hands_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "rec.csv"
    code = main(
        ["record", "--label", "wave", "--source", "test", "--width", "160", "--height", "120",
         "--max-frames", "3", "--no-window", "--model-path", str(LOCAL_MODEL), "--output", str(out)]
    )
    assert code == 1
    assert "no samples recorded" in capsys.readouterr().err
    assert not out.exists()
