"""Download and verify the MediaPipe hand landmarker model.

The model (~8 MB) is never committed to the repository. It is fetched once
into ``models/`` (or a custom path) and verified with a SHA-256 checksum.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_FILENAME = "hand_landmarker.task"
MODEL_ENV_VAR = "GESTURE_PILOT_MODEL"

_CHUNK = 64 * 1024

ProgressCallback = Callable[[int, int], None]


class ModelError(RuntimeError):
    """Raised when the model cannot be downloaded or fails verification."""


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    size: int
    downloaded: bool  # False when an existing valid file was reused


def default_model_path() -> Path:
    """Model location: ``$GESTURE_PILOT_MODEL`` or ``./models/hand_landmarker.task``."""
    env = os.environ.get(MODEL_ENV_VAR)
    if env:
        return Path(env)
    return Path("models") / MODEL_FILENAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, expected_sha256: str = MODEL_SHA256) -> bool:
    """Return True when ``path`` exists and matches the expected checksum."""
    path = Path(path)
    return path.is_file() and sha256_file(path) == expected_sha256.lower()


def download_model(
    dest: Path | None = None,
    *,
    url: str | None = None,
    expected_sha256: str | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    timeout: float = 60.0,
) -> DownloadResult:
    """Download the model to ``dest`` and verify it.

    An existing file with the right checksum is reused unless ``force`` is set.
    The download goes to a temporary file in the same directory and is only
    moved into place after the checksum matches, so a broken download never
    leaves a corrupt model behind.
    """
    dest = Path(dest) if dest is not None else default_model_path()
    url = url or MODEL_URL
    expected = (expected_sha256 or MODEL_SHA256).lower()

    if not force and dest.is_file():
        digest = sha256_file(dest)
        if digest == expected:
            return DownloadResult(dest, digest, dest.stat().st_size, downloaded=False)

    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=dest.parent)
    tmp = Path(tmp_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            try:
                response = urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
            except OSError as exc:
                raise ModelError(f"could not download model from {url}: {exc}") from exc
            with response:
                total = int(response.headers.get("Content-Length") or 0)
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if progress is not None:
                        progress(size, total)
        actual = digest.hexdigest()
        if actual != expected:
            raise ModelError(
                f"checksum mismatch for downloaded model: expected {expected}, got {actual}"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return DownloadResult(dest, actual, size, downloaded=True)
