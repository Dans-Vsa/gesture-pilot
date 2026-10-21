# gesture-pilot

Control your computer with hand gestures from a webcam.

[![CI](https://github.com/Dans-Vsa/gesture-pilot/actions/workflows/ci.yml/badge.svg)](https://github.com/Dans-Vsa/gesture-pilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`gesture-pilot` tracks your hand in real time with [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
and [OpenCV](https://opencv.org/), recognises gestures (rule-based and trainable),
and maps them to actions such as play/pause, volume, mouse movement and scrolling.
Everything runs locally on the CPU — no cloud APIs, no API keys.

> **Status:** v1.0.0 — all four milestones are done. See the [CHANGELOG](CHANGELOG.md).

## Features

- [x] `download-model` — downloads the MediaPipe hand landmarker model (~8 MB) on demand and verifies its SHA-256 checksum (the model is never committed)
- [x] Frame sources — webcam, video file, still image, or a synthetic `test` pattern (works without a camera)
- [x] `preview` — live feed with a smoothed FPS overlay, headless mode and snapshots
- [x] Hand tracking — MediaPipe `HandLandmarker` in VIDEO mode, up to N hands, 21 landmarks each, skeleton + handedness overlay (`track`)
- [x] Landmark normalisation — wrist origin, palm-size scale, left hands mirrored to right, so rules and classifiers ignore position, size and handedness
- [x] Rule-based gestures — open palm, fist, point, peace, thumbs up, thumbs down, OK, pinch (`recognize`)
- [x] Temporal smoothing — landmark moving average, debounce (gesture must be held for N frames), no repeats while held, cooldown between events
- [x] Trainable gestures — record your own labelled samples to CSV and train a scikit-learn classifier (SVM, k-NN or random forest) with a hold-out report and confusion matrix (`record`, `train`, `--model`)
- [x] Synthetic demo data — `record --synthetic N` generates samples of the built-in gestures, so you can try training without a camera
- [x] Actions — TOML mapping from gestures to media keys, volume, key combos, clicks, scrolling and cursor movement (`init-config`, `run`)
- [x] Safety — actions start disarmed, a toggle gesture arms/disarms them, optional left/right hand filter, `--dry-run` prints instead of acting
- [x] Performance options — detect on every Nth frame (`--process-every`) and downscale frames before detection (`--detect-width`)
- [x] HUD — on-screen panel with mode, gestures per hand, armed state and the last action (`--no-hud` to hide)
- [x] CI — GitHub Actions on Linux + Windows, Python 3.10–3.13, no camera needed

## Tech stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ (developed on 3.12; use a version MediaPipe ships wheels for — not 3.14 yet) |
| Vision | OpenCV (capture, drawing), MediaPipe Tasks `HandLandmarker` |
| Numerics / ML | NumPy, scikit-learn (custom gesture classifier), joblib (model files) |
| Actions | pynput (keyboard + mouse), TOML config (`tomllib`) |
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

On headless Debian/Ubuntu machines MediaPipe and OpenCV also need a few system
libraries: `sudo apt-get install libegl1 libgles2 libgl1 libglib2.0-0`.

The model is saved to `models/hand_landmarker.task`. Use `--path` or the
`GESTURE_PILOT_MODEL` environment variable to store it somewhere else.

## Demo walkthrough

A five-minute tour, from nothing to controlling your media player:

```bash
gesture-pilot download-model                 # 1. fetch the ~8 MB hand model once
gesture-pilot preview                        # 2. check the camera works (q / Esc quits)
gesture-pilot track                          # 3. see the 21-point hand skeleton
gesture-pilot recognize                      # 4. show open palm, fist, peace, thumbs up ...
gesture-pilot run --dry-run                  # 5. see which actions *would* fire
gesture-pilot run                            # 6. for real: hold "ok" to arm, then open palm = play/pause
```

The HUD in the bottom-left corner shows what the program is doing:

```
mode: run  [detect 1/2, 320x240]     <- command and performance options
ARMED                                 <- actions armed (green) or disarmed (grey)
Right: open_palm                      <- per-frame gesture of each hand
last: open_palm -> key:media_play_pause   <- last event/action, highlighted for 1.5 s
```

No webcam? Every command accepts `--source test` (synthetic pattern), a video
file or an image, plus `--no-window --snapshot out.png` to save the annotated
last frame.

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

### Custom gestures (record + train)

Teach gesture-pilot your own gestures in three steps:

```bash
# 1. Record ~200 samples per gesture (3 s countdown, then hold the pose and move it around a bit)
gesture-pilot record --label rock_on
gesture-pilot record --label call_me
gesture-pilot record --label fist          # include the built-in poses you also want to keep

# 2. Train a classifier (prints precision/recall per gesture and a confusion matrix)
gesture-pilot train                        # data/gestures.csv -> models/gestures.joblib

# 3. Use it instead of the rule-based recogniser
gesture-pilot recognize --model models/gestures.joblib
```

Samples are stored as normalised landmarks (63 numbers per hand), so position,
size and left/right hand do not matter. `--algorithm svm|knn|forest` picks the
classifier (default: calibrated RBF SVM); predictions below `--min-score`
(default 0.6) count as `none`. Each gesture needs at least 10 samples.

No camera at hand? Generate a demo dataset of the eight built-in gestures:

```bash
gesture-pilot record --synthetic 100 --output data/demo.csv
gesture-pilot train data/demo.csv --output models/demo.joblib
```

> Model files are joblib pickles — only load models you trained yourself.

### Actions

```bash
gesture-pilot run --dry-run            # print what would happen, touch nothing
gesture-pilot init-config              # write gestures.toml to customise the mapping
gesture-pilot run --config gestures.toml
gesture-pilot run --config gestures.toml --model models/gestures.joblib
```

Actions start **disarmed**. Hold the toggle gesture (default: `ok`) to arm them,
and again to disarm; the overlay shows `ARMED` / `disarmed`. The default mapping:

| Gesture | Action |
| --- | --- |
| `ok` | `toggle` — arm / disarm |
| `open_palm` | play / pause |
| `thumbs_up` / `thumbs_down` | volume up / down |
| `peace` | next track |
| `point` | move the cursor with your index finger |
| `pinch` | left click |

`gestures.toml` format:

```toml
[settings]
start_armed = false   # or pass --armed
stable_frames = 5     # frames a gesture must be held before it fires
cooldown = 0.6        # seconds between events
hand = "any"          # "Left" / "Right": only this hand triggers actions

[actions]
ok = "toggle"
open_palm = "key:media_play_pause"
peace = "key:ctrl+shift+t"
fist = "scroll:-3"
pinch = "click:left"
point = "move:1500"
rock_on = "key:f5"    # custom gestures from `train` work too
```

| Action | Meaning |
| --- | --- |
| `key:<keys>` | press a key or combo joined with `+`: letters/digits, `ctrl`, `shift`, `alt`, `cmd`, `enter`, `space`, `tab`, `esc`, arrows, `f1`–`f12`, `media_play_pause`, `media_next`, `media_previous`, `media_volume_up/down/mute` (aliases: `volume_up`, `mute`, `play_pause`, …) |
| `click[:button[:count]]` | mouse click — `left` (default), `right`, `middle`; count 1–3 |
| `scroll:<dy>` / `scroll:<dx>,<dy>` | scroll by wheel steps (positive = up / right) |
| `move[:speed]` | while held, the cursor follows the index fingertip (relative; `speed` = pixels per frame width, default 1500) |
| `toggle` | arm / disarm the other actions (at most one; with none, actions are always armed) |

### Performance

On a slow laptop, trade a little accuracy for speed:

| Option | Default | Meaning |
| --- | --- | --- |
| `--process-every N` | 1 | run hand detection on every Nth frame and reuse the result in between |
| `--detect-width PX` | full size | downscale frames to this width before detection (min 160); landmarks still map onto the full frame |
| `--width` / `--height` | camera default | requested capture resolution |
| `--max-hands N` | 2 | fewer hands = faster detection |
| `--no-hud` | off | hide the status panel |

```bash
gesture-pilot run --process-every 2 --detect-width 320 --max-hands 1
```

With `--process-every 2` gestures need twice as many camera frames to become
stable, so you may want a lower `--stable-frames` (or `stable_frames` in the config).

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

CI runs the same suite on Linux and Windows for Python 3.10–3.13 (with the
model downloaded, so the tracker tests run too) plus a few CLI smoke tests.

The tests never open a camera or a window and never touch the network: the
model downloader is tested against local `file://` URLs, the sources use
generated videos, images and the test pattern, and gesture rules are tested on
synthetic hands from `gesture_pilot.synthetic` (with random rotation, scale,
position, handedness and noise). Classifiers are trained on generated
datasets, and actions run against a recording `DryRunBackend` — tests never
press real keys or move the mouse. Tests that need the real model are skipped
when `models/hand_landmarker.task` has not been downloaded.

## Project structure

```
gesture-pilot/
├── src/gesture_pilot/
│   ├── cli.py        # argparse CLI: download-model, preview, track, recognize,
│   │                 #   record, train, init-config, run
│   ├── model.py      # model download + SHA-256 verification
│   ├── sources.py    # camera / video / image / test-pattern frame sources
│   ├── tracker.py    # MediaPipe HandLandmarker wrapper (VIDEO mode)
│   ├── landmarks.py  # Hand dataclass, landmark indices, normalisation
│   ├── gestures.py   # finger states + rule-based gesture classifier
│   ├── smoothing.py  # landmark smoothing, gesture debounce/cooldown
│   ├── dataset.py    # labelled samples CSV, synthetic demo data
│   ├── classifier.py # scikit-learn gesture classifier (train/save/load/predict)
│   ├── actions.py    # action parsing, dry-run + pynput backends
│   ├── config.py     # TOML gesture -> action mapping
│   ├── controller.py # events -> actions, safety toggle, cursor movement
│   ├── synthetic.py  # synthetic hand generator (tests, demo data)
│   ├── perf.py       # frame skipping, downscaling before detection
│   ├── hud.py        # on-screen status panel
│   ├── fps.py        # smoothed FPS counter
│   └── overlay.py    # labels, FPS badge, hand skeleton
├── tests/            # pytest suite (no camera required)
├── .github/workflows/ci.yml  # tests on Linux + Windows, Python 3.10–3.13
├── models/           # downloaded / trained models (git-ignored)
├── data/             # recorded datasets (git-ignored)
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

## Roadmap

1. ✅ Scaffold, model downloader, frame sources, `preview`
2. ✅ Hand tracking, landmark normalisation, rule-based gestures with smoothing/debounce (`track`, `recognize`)
3. ✅ Trainable gestures (`record`, `train`) and gesture → action mappings (`init-config`, `run`, `--dry-run`)
4. ✅ Polish: performance options, HUD, CI, CHANGELOG, v1.0.0

Future ideas: two-hand gestures (e.g. zoom with both hands), dynamic gestures
(swipes) from landmark trajectories, a small GUI for editing the mapping.

## License

[MIT](LICENSE) © Dans-Vsa
