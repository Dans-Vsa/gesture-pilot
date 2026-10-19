# gesture-pilot

Control your computer with hand gestures from a webcam.

`gesture-pilot` tracks your hand in real time with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
and [OpenCV](https://opencv.org/), recognises gestures (rule-based and trainable),
and maps them to actions such as play/pause, volume, mouse movement and scrolling.
Everything runs locally on the CPU — no cloud APIs, no API keys.

> **Status:** in development (v0.2.0). Milestone 2 of 4 is done — see the roadmap below.

## Features

- [x] `download-model` — downloads the MediaPipe hand landmarker model (~8 MB) on demand and verifies its SHA-256 checksum (the model is never committed)
- [x] Frame sources — webcam, video file, still image, or a synthetic `test` pattern (works without a camera)
- [x] `preview` — live feed with a smoothed FPS overlay, headless mode and snapshots
- [x] Hand tracking — MediaPipe `HandLandmarker` in VIDEO mode, up to N hands, 21 landmarks each, skeleton + handedness overlay (`track`)
- [x] Landmark normalisation — wrist origin, palm-size scale, left hands mirrored to right, so rules and classifiers ignore position, size and handedness
- [x] Rule-based gestures — open palm, fist, point, peace, thumbs up, thumbs down, OK, pinch (`recognize`)
- [x] Temporal smoothing — landmark moving average, debounce (gesture must be held for N frames), no repeats while held, cooldown between events
- [ ] Trainable gestures — record your own labelled samples and train a scikit-learn classifier (`record`, `train`)
- [ ] Actions — TOML mapping from gestures to media keys, volume, key combos, mouse and scroll, with a safety gesture and `--dry-run` (`run`)
- [ ] Polish — performance tuning, HUD, CI on Linux + Windows, v1.0.0

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ (developed on 3.12; use a version MediaPipe ships wheels for — not 3.14 yet) |
| Vision | OpenCV (capture, drawing), MediaPipe Tasks `HandLandmarker` |
| Numerics / ML | NumPy (scikit-learn planned for custom gestures in milestone 3) |
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
`--no-mirror` to disable that. Mirroring happens before detection, so the
reported handedness matches your real hand.

### Hand tracking

```bash
# Draw the 21-landmark skeleton of up to two hands
gesture-pilot track

# One hand, stricter confidence, custom model location
gesture-pilot track --max-hands 1 --min-confidence 0.7 --model-path D:/models/hand_landmarker.task

# Run on a recorded clip without a window and keep the last frame
gesture-pilot track --source clip.mp4 --no-window --snapshot tracked.png
```

### Gesture recognition

```bash
gesture-pilot recognize
```

Hold a gesture steadily and an event is printed (time since start, hand, gesture):

```
    3.41s  Right  open_palm
    5.02s  Right  fist
    7.88s  Left   thumbs_up
```

| Gesture | How to make it |
| --- | --- |
| `open_palm` | all five fingers extended |
| `fist` | all fingers curled |
| `point` | only the index finger extended |
| `peace` | index and middle fingers extended |
| `thumbs_up` / `thumbs_down` | only the thumb extended, pointing up / down |
| `ok` | thumb and index tips touching, other three fingers extended |
| `pinch` | thumb and index tips touching, other fingers curled |

Tuning options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--stable-frames N` | 5 | frames a gesture must be held before it fires |
| `--cooldown S` | 0.5 | seconds after an event during which nothing else fires |
| `--smoothing A` | 0.6 | landmark moving-average weight in (0, 1]; 1 = off |

A gesture fires once while it is held; release it (or switch to another one)
to fire again. The overlay shows the per-frame gesture of each hand, with `*`
when it is the currently held (fired) gesture.

### How the rules work

1. Landmarks are normalised: wrist → origin, divided by the palm size
   (wrist → middle-finger knuckle), y flipped to point up, left hands mirrored.
2. A finger counts as *extended* when it is nearly straight (bend angle below
   55°) and its tip is farther from the wrist than its middle joint. The thumb
   uses its tip's distance from the pinky knuckle instead.
3. Thumb–index tip distance below 0.25 palm sizes means the tips touch
   (`ok` / `pinch`); the direction of the thumb separates thumbs up and down.

## Tests

```bash
python -m pytest
```

The tests never open a camera or a window and never touch the network: the
model downloader is tested against local `file://` URLs, the sources use
generated videos, images and the test pattern, and gesture rules are tested on
synthetic hands from `gesture_pilot.synthetic` (with random rotation, scale,
position, handedness and noise). Tests that need the real model are skipped
when `models/hand_landmarker.task` has not been downloaded.

## Project structure

```
gesture-pilot/
├── src/gesture_pilot/
│   ├── cli.py        # argparse CLI: download-model, preview, track, recognize
│   ├── model.py      # model download + SHA-256 verification
│   ├── sources.py    # camera / video / image / test-pattern frame sources
│   ├── tracker.py    # MediaPipe HandLandmarker wrapper (VIDEO mode)
│   ├── landmarks.py  # Hand dataclass, landmark indices, normalisation
│   ├── gestures.py   # finger states + rule-based gesture classifier
│   ├── smoothing.py  # landmark smoothing, gesture debounce/cooldown
│   ├── synthetic.py  # synthetic hand generator (tests, demo data)
│   ├── fps.py        # smoothed FPS counter
│   └── overlay.py    # labels, FPS badge, hand skeleton
├── tests/            # pytest suite (no camera required)
├── models/           # downloaded model (git-ignored)
├── pyproject.toml
└── LICENSE
```

## Roadmap

1. ✅ Scaffold, model downloader, frame sources, `preview`
2. ✅ Hand tracking, landmark normalisation, rule-based gestures with smoothing/debounce (`track`, `recognize`)
3. Trainable gestures (`record`, `train`) and gesture → action mappings (`run`, `--dry-run`)
4. Polish: performance options, HUD, CI, CHANGELOG, v1.0.0

## License

[MIT](LICENSE) © Dans-Vsa
