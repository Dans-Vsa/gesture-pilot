# gesture-pilot

Control your computer with hand gestures from a webcam.

`gesture-pilot` tracks your hand in real time with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
and [OpenCV](https://opencv.org/), recognises gestures (rule-based and trainable),
and maps them to actions such as play/pause, volume, mouse movement and scrolling.
Everything runs locally on the CPU — no cloud APIs, no API keys.

> **Status:** early development (v0.1.0). Milestone 1 of 6 is done — see the roadmap below.

## Features

- [x] `download-model` — downloads the MediaPipe hand landmarker model (~8 MB) on demand and verifies its SHA-256 checksum (the model is never committed)
- [x] Frame sources — webcam, video file, still image, or a synthetic `test` pattern (works without a camera)
- [x] `preview` — live feed with a smoothed FPS overlay, headless mode and snapshots
- [ ] Hand tracking — 21 landmarks per hand, normalised (wrist origin, palm-size scale, handedness), skeleton drawing (`track`)
- [ ] Rule-based gestures — open palm, fist, point, peace, thumbs up/down, OK, pinch; temporal smoothing and debounce (`recognize`)
- [ ] Trainable gestures — record your own labelled samples and train a scikit-learn classifier (`record`, `train`)
- [ ] Actions — TOML mapping from gestures to media keys, volume, key combos, mouse and scroll, with a safety gesture and `--dry-run` (`run`)
- [ ] Polish — performance tuning, HUD, CI on Linux + Windows, v1.0.0

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ (developed on 3.12; use a version MediaPipe ships wheels for — not 3.14 yet) |
| Vision | OpenCV (capture, drawing), MediaPipe Tasks `HandLandmarker` |
| Numerics / ML | NumPy (scikit-learn planned for custom gestures) |
| Actions | pynput (planned) |
| Tests | pytest — no camera needed |

## Setup

```bash
git clone https://github.com/Dans-Vsa/gesture-pilot.git
cd gesture-pilot
py -3.12 -m venv .venv            # Windows; on Linux/macOS: python3.12 -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
gesture-pilot download-model
```

The model is saved to `models/hand_landmarker.task`. Use `--path` or the
`GESTURE_PILOT_MODEL` environment variable to store it somewhere else.

## Usage

```bash
gesture-pilot --help

# Live webcam preview with FPS overlay (q or Esc to quit)
gesture-pilot preview

# Second camera at 1280x720
gesture-pilot preview --source 1 --width 1280 --height 720

# No camera? Use the synthetic test pattern, a video or an image
gesture-pilot preview --source test
gesture-pilot preview --source clip.mp4

# Headless (e.g. on a server): process 100 frames and save the last one
gesture-pilot preview --source test --max-frames 100 --no-window --snapshot last.png
```

Camera frames are mirrored horizontally by default (like a selfie view); pass
`--no-mirror` to disable that.

## Tests

```bash
python -m pytest
```

The tests never open a camera or a window and never touch the network: the
model downloader is tested against local `file://` URLs, and the sources use
generated videos, images and the test pattern.

## Project structure

```
gesture-pilot/
├── src/gesture_pilot/
│   ├── cli.py        # argparse CLI: download-model, preview
│   ├── model.py      # model download + SHA-256 verification
│   ├── sources.py    # camera / video / image / test-pattern frame sources
│   ├── fps.py        # smoothed FPS counter
│   └── overlay.py    # on-screen labels and FPS badge
├── tests/            # pytest suite (no camera required)
├── models/           # downloaded model (git-ignored)
├── pyproject.toml
└── LICENSE
```

## Roadmap

1. ✅ Scaffold, model downloader, frame sources, `preview`
2. Hand tracking and landmark normalisation (`track`)
3. Rule-based gestures with smoothing/debounce (`recognize`)
4. Trainable gestures (`record`, `train`)
5. Gesture → action mappings (`run`, `--dry-run`)
6. Polish, CI, v1.0.0

## License

[MIT](LICENSE) © Dans-Vsa
