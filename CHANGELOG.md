# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-21

### Added
- `--process-every N` frame skipping: hand detection runs on every Nth frame and
  the last result is reused in between, so the overlay and actions keep the full frame rate.
- `--detect-width PX` downscales frames before detection (landmarks map back to the
  full frame automatically).
- On-screen HUD with the mode, performance options, per-hand gestures, armed state and
  the last event/action (highlighted for 1.5 s); hide it with `--no-hud`.
- GitHub Actions CI on Linux and Windows, Python 3.10–3.13, with a cached, checksum-verified
  model download and CLI smoke tests.
- This changelog and a demo walkthrough in the README.

### Changed
- `record` only stores samples from frames that went through detection, so frame
  skipping never produces duplicate samples.
- The summary line reports how many frames were detected when frame skipping is on.

## [0.3.0] - 2026-10-20

### Added
- `record` (labelled samples to CSV, `--synthetic N` demo data) and `train`
  (scikit-learn SVM / k-NN / random forest with hold-out report and confusion matrix).
- `--model` / `--min-score` to use a trained classifier in `recognize` and `run`.
- Gesture → action mapping in TOML (`init-config`): media keys, key combos, clicks,
  scrolling and relative cursor movement via pynput.
- `run` command with a safety toggle gesture, `--armed`, `--dry-run` and a left/right hand filter.

## [0.2.0] - 2026-10-19

### Added
- MediaPipe `HandLandmarker` tracking in VIDEO mode and skeleton overlay (`track`).
- Landmark normalisation (wrist origin, palm-size scale, mirrored left hands).
- Rule-based gestures: open palm, fist, point, peace, thumbs up/down, OK, pinch (`recognize`).
- Landmark smoothing, debounce and cooldown for gesture events.

## [0.1.0] - 2026-10-18

### Added
- Project scaffold, MIT license, `gesture-pilot` console script.
- `download-model` with SHA-256 verification.
- Camera / video / image / test-pattern frame sources and `preview` with an FPS overlay.

[1.0.0]: https://github.com/Dans-Vsa/gesture-pilot/releases/tag/v1.0.0
