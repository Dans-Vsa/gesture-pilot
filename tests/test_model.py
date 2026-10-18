import hashlib
from pathlib import Path

import pytest

from gesture_pilot import model
from gesture_pilot.model import ModelError, default_model_path, download_model, verify_model

PAYLOAD = b"fake model bytes " * 1000
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def remote(tmp_path: Path) -> str:
    """A file:// URL serving a fake model."""
    src = tmp_path / "remote" / "hand_landmarker.task"
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    return src.as_uri()


def test_download_and_verify(tmp_path: Path, remote: str) -> None:
    dest = tmp_path / "models" / "hand.task"
    seen: list[int] = []
    result = download_model(dest, url=remote, expected_sha256=PAYLOAD_SHA, progress=lambda d, t: seen.append(d))
    assert result.downloaded
    assert result.size == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert verify_model(dest, PAYLOAD_SHA)
    assert seen and seen[-1] == len(PAYLOAD)
    assert not list(dest.parent.glob("*.part"))


def test_existing_valid_file_is_reused(tmp_path: Path, remote: str) -> None:
    dest = tmp_path / "hand.task"
    dest.write_bytes(PAYLOAD)
    result = download_model(dest, url="file:///does/not/exist", expected_sha256=PAYLOAD_SHA)
    assert not result.downloaded


def test_force_downloads_again(tmp_path: Path, remote: str) -> None:
    dest = tmp_path / "hand.task"
    dest.write_bytes(PAYLOAD)
    assert download_model(dest, url=remote, expected_sha256=PAYLOAD_SHA, force=True).downloaded


def test_corrupt_existing_file_is_replaced(tmp_path: Path, remote: str) -> None:
    dest = tmp_path / "hand.task"
    dest.write_bytes(b"truncated")
    assert download_model(dest, url=remote, expected_sha256=PAYLOAD_SHA).downloaded
    assert dest.read_bytes() == PAYLOAD


def test_checksum_mismatch_keeps_no_file(tmp_path: Path, remote: str) -> None:
    dest = tmp_path / "hand.task"
    with pytest.raises(ModelError, match="checksum mismatch"):
        download_model(dest, url=remote, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_unreachable_url_raises_model_error(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="could not download"):
        download_model(tmp_path / "hand.task", url=(tmp_path / "missing").as_uri())


def test_verify_missing_file(tmp_path: Path) -> None:
    assert not verify_model(tmp_path / "nope.task")


def test_default_path_and_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(model.MODEL_ENV_VAR, raising=False)
    assert default_model_path() == Path("models") / model.MODEL_FILENAME
    monkeypatch.setenv(model.MODEL_ENV_VAR, str(tmp_path / "custom.task"))
    assert default_model_path() == tmp_path / "custom.task"
