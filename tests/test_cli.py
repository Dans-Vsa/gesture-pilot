import hashlib
from pathlib import Path

import cv2
import pytest

from gesture_pilot import __version__, model
from gesture_pilot.cli import main


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "download-model" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_preview_headless_test_pattern(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    snap = tmp_path / "out" / "last.png"
    code = main(
        ["preview", "--source", "test", "--width", "160", "--height", "120",
         "--max-frames", "5", "--no-window", "--snapshot", str(snap)]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "test pattern: 5 frames, 160x120" in out
    image = cv2.imread(str(snap))
    assert image is not None and image.shape == (120, 160, 3)


def test_preview_missing_file_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["preview", "--source", str(tmp_path / "nope.mp4"), "--no-window"])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_preview_rejects_non_positive_frames() -> None:
    with pytest.raises(SystemExit):
        main(["preview", "--max-frames", "0"])


def test_download_model_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = b"model" * 100
    remote = tmp_path / "remote.task"
    remote.write_bytes(payload)
    monkeypatch.setattr(model, "MODEL_URL", remote.as_uri())
    monkeypatch.setattr(model, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    dest = tmp_path / "models" / "hand.task"

    assert main(["download-model", "--path", str(dest), "--quiet"]) == 0
    assert "model downloaded" in capsys.readouterr().out
    assert main(["download-model", "--path", str(dest), "--quiet"]) == 0
    assert "already present" in capsys.readouterr().out


def test_download_model_command_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(model, "MODEL_URL", (tmp_path / "missing").as_uri())
    assert main(["download-model", "--path", str(tmp_path / "m.task"), "--quiet"]) == 1
    assert "could not download" in capsys.readouterr().err
