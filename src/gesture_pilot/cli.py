"""Command-line interface: ``gesture-pilot <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from gesture_pilot import __version__

WINDOW_TITLE = "gesture-pilot"
EXIT_KEYS = {27, ord("q")}  # Esc, q


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesture-pilot",
        description="Control your computer with hand gestures from a webcam.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    dl = sub.add_parser(
        "download-model",
        help="download and verify the MediaPipe hand landmarker model",
        description="Download the MediaPipe hand landmarker model (~8 MB) and verify its SHA-256.",
    )
    dl.add_argument("--path", type=Path, help="where to store the model (default: models/hand_landmarker.task)")
    dl.add_argument("--force", action="store_true", help="download again even if a valid file exists")
    dl.add_argument("--quiet", action="store_true", help="no progress output")

    pv = sub.add_parser(
        "preview",
        help="show the camera/video feed with an FPS overlay",
        description="Show frames from a camera, video, image or test pattern with an FPS overlay. "
        "Press q or Esc to quit.",
    )
    _add_source_args(pv)
    pv.add_argument("--no-window", action="store_true", help="headless: process frames without a window")
    pv.add_argument("--snapshot", type=Path, help="save the last processed frame to this image file")

    tr = sub.add_parser(
        "track",
        help="track hands and draw the 21-landmark skeleton",
        description="Detect and track hands with MediaPipe and draw their skeletons. "
        "Press q or Esc to quit.",
    )
    _add_source_args(tr)
    _add_tracker_args(tr)

    rc = sub.add_parser(
        "recognize",
        help="recognise gestures and print debounced gesture events",
        description="Track hands, recognise rule-based gestures (open palm, fist, point, peace, "
        "thumbs up/down, OK, pinch) and print an event whenever a gesture is held steadily. "
        "Press q or Esc to quit.",
    )
    _add_source_args(rc)
    _add_tracker_args(rc)
    _add_classifier_args(rc)
    rc.add_argument(
        "--stable-frames", type=_positive_int, default=5,
        help="frames a gesture must be held before it fires (default: 5)",
    )
    rc.add_argument(
        "--cooldown", type=_non_negative_float, default=0.5,
        help="seconds after an event during which no other event fires (default: 0.5)",
    )
    _add_smoothing_arg(rc)

    rec = sub.add_parser(
        "record",
        help="record labelled gesture samples to a CSV dataset",
        description="Track one hand and append its normalised landmarks, labelled with --label, "
        "to a CSV dataset for `train`. Use --synthetic to generate demo data for the built-in "
        "gestures without a camera.",
    )
    rec.add_argument("--label", help="gesture name for the recorded samples, e.g. rock_on")
    rec.add_argument(
        "--output", type=Path, default=Path("data/gestures.csv"),
        help="dataset file to append to (default: data/gestures.csv)",
    )
    rec.add_argument("--samples", type=_positive_int, default=200, help="samples to record (default: 200)")
    rec.add_argument("--every", type=_positive_int, default=2, help="keep every Nth frame (default: 2)")
    rec.add_argument(
        "--delay", type=_non_negative_float, default=3.0,
        help="seconds to get ready before recording from a camera (default: 3)",
    )
    rec.add_argument(
        "--synthetic", type=_positive_int, metavar="N",
        help="no camera: write N synthetic samples per built-in gesture instead",
    )
    rec.add_argument("--seed", type=int, default=0, help="random seed for --synthetic (default: 0)")
    _add_source_args(rec)
    _add_tracker_args(rec)
    _add_smoothing_arg(rec)

    trn = sub.add_parser(
        "train",
        help="train a custom gesture classifier from a CSV dataset",
        description="Train a scikit-learn classifier on samples recorded with `record`, print a "
        "hold-out evaluation and save the model (refitted on all samples) with joblib.",
    )
    trn.add_argument("dataset", type=Path, nargs="?", default=Path("data/gestures.csv"),
                     help="dataset CSV (default: data/gestures.csv)")
    trn.add_argument("--output", type=Path, default=Path("models/gestures.joblib"),
                     help="where to save the model (default: models/gestures.joblib)")
    trn.add_argument("--algorithm", choices=("svm", "knn", "forest"), default="svm",
                     help="classifier (default: svm)")
    trn.add_argument("--test-size", type=float, default=0.2, help="hold-out fraction (default: 0.2)")
    trn.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")

    ic = sub.add_parser(
        "init-config",
        help="write an example gesture -> action config (TOML)",
        description="Write the default gesture -> action mapping to a TOML file you can edit.",
    )
    ic.add_argument("path", type=Path, nargs="?", default=Path("gestures.toml"),
                    help="file to create (default: gestures.toml)")
    ic.add_argument("--force", action="store_true", help="overwrite an existing file")

    run = sub.add_parser(
        "run",
        help="control the computer: map gestures to keys, mouse and scrolling",
        description="Recognise gestures and perform the mapped actions (media keys, key combos, "
        "clicks, scrolling, cursor movement). Actions start disarmed: hold the toggle gesture "
        "(default: ok) to arm them. Use --dry-run to only print what would happen. "
        "Press q or Esc to quit.",
    )
    run.add_argument("--config", type=Path, help="gesture -> action TOML (default: built-in mapping)")
    run.add_argument("--dry-run", action="store_true", help="print actions instead of performing them")
    run.add_argument("--armed", action="store_true", help="start armed (ignore start_armed in the config)")
    _add_source_args(run)
    _add_tracker_args(run)
    _add_classifier_args(run)
    _add_smoothing_arg(run)
    return parser


def _add_smoothing_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smoothing", type=_alpha, default=0.6,
        help="landmark smoothing factor in (0, 1]; 1 disables smoothing (default: 0.6)",
    )


def _add_classifier_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", type=Path,
        help="custom gesture classifier from `train` (default: built-in rule-based gestures)",
    )
    parser.add_argument(
        "--min-score", type=_confidence, default=0.6,
        help="minimum classifier confidence, otherwise 'none' (default: 0.6)",
    )


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return number


def _alpha(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return number


def _confidence(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _add_tracker_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path", type=Path,
        help="hand landmarker model (default: $GESTURE_PILOT_MODEL or models/hand_landmarker.task)",
    )
    parser.add_argument("--max-hands", type=_positive_int, default=2, help="hands to track (default: 2)")
    parser.add_argument(
        "--min-confidence", type=_confidence, default=0.5,
        help="minimum detection/tracking confidence (default: 0.5)",
    )
    parser.add_argument("--no-window", action="store_true", help="headless: process frames without a window")
    parser.add_argument("--snapshot", type=Path, help="save the last processed frame to this image file")


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        default="0",
        help="camera index (0, 1, ...), video file, image file, or 'test' for a synthetic pattern "
        "(default: 0)",
    )
    parser.add_argument("--width", type=_positive_int, help="requested frame width")
    parser.add_argument("--height", type=_positive_int, help="requested frame height")
    parser.add_argument("--max-frames", type=_positive_int, help="stop after this many frames")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror camera frames horizontally")


def cmd_download_model(args: argparse.Namespace) -> int:
    from gesture_pilot.model import ModelError, default_model_path, download_model

    dest = args.path or default_model_path()
    shown = {"pct": -1}

    def progress(done: int, total: int) -> None:
        if total:
            pct = done * 100 // total
            if pct - shown["pct"] < 5 and pct != 100:
                return
            shown["pct"] = pct
            print(f"\rdownloading {dest.name}: {pct:3d}% ({done / 1e6:.1f} MB)", end="", file=sys.stderr)

    try:
        result = download_model(dest, force=args.force, progress=None if args.quiet else progress)
    except ModelError as exc:
        if not args.quiet:
            print(file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.downloaded and not args.quiet:
        print(file=sys.stderr)
    state = "downloaded" if result.downloaded else "already present"
    print(f"model {state}: {result.path} ({result.size / 1e6:.1f} MB, sha256 {result.sha256[:12]}...)")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    import cv2

    from gesture_pilot.fps import FPSCounter
    from gesture_pilot.overlay import draw_fps
    from gesture_pilot.sources import CameraSource, SourceError, open_source

    try:
        source = open_source(args.source, width=args.width, height=args.height, max_frames=args.max_frames)
    except SourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mirror = isinstance(source, CameraSource) and not args.no_mirror
    counter = FPSCounter()
    last = None
    with source:
        for frame in source:
            if mirror:
                frame = cv2.flip(frame, 1)
            draw_fps(frame, counter.tick())
            last = frame
            if not args.no_window:
                cv2.imshow(WINDOW_TITLE, frame)
                if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS:
                    break
            if args.max_frames and counter.frames >= args.max_frames:
                break
    if not args.no_window:
        cv2.destroyAllWindows()

    if last is None:
        print(f"error: no frames received from {source.name}", file=sys.stderr)
        return 1
    if args.snapshot and not _save_snapshot(args.snapshot, last):
        return 1
    height, width = last.shape[:2]
    print(
        f"{source.name}: {counter.frames} frames, {width}x{height}, "
        f"average {counter.average:.1f} FPS"
    )
    return 0


def _save_snapshot(path: Path, frame: object) -> bool:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        print(f"error: could not write snapshot {path}", file=sys.stderr)
        return False
    return True


def _make_labeler(args: argparse.Namespace):
    """Per-hand gesture function: the trained classifier with --model, else the rules."""
    from gesture_pilot.gestures import classify

    if getattr(args, "model", None) is None:
        return lambda hand: classify(hand).gesture
    from gesture_pilot.classifier import GestureClassifier

    classifier = GestureClassifier.load(args.model, min_confidence=args.min_score)
    return lambda hand: classifier.predict(hand).gesture


def _run_tracking(args: argparse.Namespace, *, mode: str) -> int:
    """Shared camera loop for ``track``, ``recognize``, ``run`` and ``record``."""
    import time

    import cv2

    from gesture_pilot.dataset import sample_row
    from gesture_pilot.fps import FPSCounter
    from gesture_pilot.gestures import NONE
    from gesture_pilot.overlay import draw_fps, draw_hand, draw_label
    from gesture_pilot.smoothing import LandmarkSmoother, MultiHandStabilizer
    from gesture_pilot.sources import CameraSource, SourceError, open_source
    from gesture_pilot.tracker import HandTracker, TrackerError

    recognize = mode in ("recognize", "run")
    labeler = controller = backend = None
    stable_frames, cooldown = getattr(args, "stable_frames", 5), getattr(args, "cooldown", 0.5)
    if recognize:
        from gesture_pilot.classifier import ClassifierError

        try:
            labeler = _make_labeler(args)
        except ClassifierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if mode == "run":
        from gesture_pilot.actions import DryRunBackend, PynputBackend
        from gesture_pilot.config import ConfigError, load_config
        from gesture_pilot.controller import GestureController

        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.armed:
            config = replace(config, start_armed=True)
        try:
            backend = DryRunBackend() if args.dry_run else PynputBackend()
        except RuntimeError as exc:
            print(f"error: {exc} (use --dry-run to only print actions)", file=sys.stderr)
            return 1
        controller = GestureController(config, backend)
        stable_frames, cooldown = config.stable_frames, config.cooldown
        toggle = config.toggle_gesture
        state = "armed" if controller.armed else f"disarmed - hold '{toggle}' to arm"
        print(f"{len(config.actions)} gesture actions loaded ({state})"
              + (", dry run" if args.dry_run else ""), flush=True)

    try:
        tracker = HandTracker(
            args.model_path,
            max_hands=args.max_hands,
            min_detection_confidence=args.min_confidence,
            min_presence_confidence=args.min_confidence,
            min_tracking_confidence=args.min_confidence,
        )
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        source = open_source(args.source, width=args.width, height=args.height, max_frames=args.max_frames)
    except SourceError as exc:
        tracker.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1

    mirror = isinstance(source, CameraSource) and not args.no_mirror
    frame_ms = 1000.0 / (source.fps or 30.0)
    smoother = LandmarkSmoother(args.smoothing) if mode != "track" else None
    stabilizer = MultiHandStabilizer(stable_frames, cooldown) if recognize else None
    counter = FPSCounter()
    start = time.monotonic()
    record_after = args.delay if mode == "record" and source.live else 0.0
    rows: list[list[str]] = []
    frames_with_hands = 0
    events = 0
    last = None

    with tracker, source:
        for index, frame in enumerate(source):
            if mirror:
                frame = cv2.flip(frame, 1)
            if source.live:
                timestamp_ms = int((time.monotonic() - start) * 1000)
            else:
                timestamp_ms = int(index * frame_ms)
            hands = tracker.detect(frame, timestamp_ms)
            if hands:
                frames_with_hands += 1
            if smoother is not None:
                hands = smoother.update(hands)

            if recognize:
                labels: dict[str, str] = {}
                for hand in hands:
                    labels.setdefault(hand.handedness, labeler(hand))
                for event in stabilizer.update(labels, timestamp_ms / 1000):
                    events += 1
                    line = f"{event.timestamp:8.2f}s  {event.handedness:<5}  {event.gesture}"
                    if controller is not None:
                        message = controller.handle_event(event)
                        if message:
                            line += f"  [{message}]"
                    print(line, flush=True)
                if controller is not None:
                    controller.update_frame(hands, labels)
                for hand in hands:
                    gesture = labels.get(hand.handedness, NONE)
                    held = stabilizer.active(hand.handedness)
                    marker = " *" if held != NONE and held == gesture else ""
                    draw_hand(frame, hand, label=f"{hand.handedness}: {gesture}{marker}")
                if controller is not None:
                    draw_label(frame, "ARMED" if controller.armed else "disarmed", (10, 70),
                               color=(80, 220, 80) if controller.armed else (160, 160, 160))
            elif mode == "record":
                waiting = timestamp_ms / 1000 < record_after
                if hands and not waiting and index % args.every == 0 and len(rows) < args.samples:
                    rows.append(sample_row(args.label, hands[0]))
                for hand in hands:
                    draw_hand(frame, hand, label=args.label)
                status = (f"get ready: {record_after - timestamp_ms / 1000:.0f}s" if waiting
                          else f"recording {args.label}: {len(rows)}/{args.samples}")
                draw_label(frame, status, (10, 70))
            else:
                for hand in hands:
                    draw_hand(frame, hand, label=f"{hand.handedness} {hand.score:.2f}")

            draw_fps(frame, counter.tick())
            if not hands:
                draw_label(frame, "no hand", (10, 44), color=(120, 120, 255))
            last = frame
            if not args.no_window:
                cv2.imshow(WINDOW_TITLE, frame)
                if (cv2.waitKey(1) & 0xFF) in EXIT_KEYS:
                    break
            if args.max_frames and counter.frames >= args.max_frames:
                break
            if mode == "record" and len(rows) >= args.samples:
                break
    if not args.no_window:
        cv2.destroyAllWindows()

    if last is None:
        print(f"error: no frames received from {source.name}", file=sys.stderr)
        return 1
    if args.snapshot and not _save_snapshot(args.snapshot, last):
        return 1
    rate = frames_with_hands * 100 / max(1, counter.frames)
    summary = (
        f"{source.name}: {counter.frames} frames, hands in {frames_with_hands} ({rate:.0f}%), "
        f"average {counter.average:.1f} FPS"
    )
    if recognize:
        summary += f", {events} gesture events"
    print(summary)
    if mode == "record":
        return _save_recording(args.output, rows)
    if mode == "run" and args.dry_run and backend.log:
        moves = sum(1 for entry in backend.log if entry.startswith("move"))
        print(f"dry run: {len(backend.log) - moves} actions, {moves} cursor moves")
    return 0


def _save_recording(path: Path, rows: list[list[str]]) -> int:
    from gesture_pilot.dataset import DatasetError, append_samples

    if not rows:
        print("error: no samples recorded (no hand detected)", file=sys.stderr)
        return 1
    try:
        written = append_samples(path, rows)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"saved {written} samples to {path}")
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    return _run_tracking(args, mode="track")


def cmd_recognize(args: argparse.Namespace) -> int:
    return _run_tracking(args, mode="recognize")


def cmd_run(args: argparse.Namespace) -> int:
    return _run_tracking(args, mode="run")


def cmd_record(args: argparse.Namespace) -> int:
    from gesture_pilot.dataset import DatasetError, append_samples, synthetic_samples, validate_label

    if args.synthetic:
        if args.label:
            print("error: --synthetic generates the built-in gestures; do not pass --label", file=sys.stderr)
            return 1
        rows = synthetic_samples(args.synthetic, seed=args.seed)
        try:
            written = append_samples(args.output, rows)
        except DatasetError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"saved {written} synthetic samples ({args.synthetic} per gesture) to {args.output}")
        return 0
    if not args.label:
        print("error: --label is required (or use --synthetic N)", file=sys.stderr)
        return 1
    try:
        validate_label(args.label)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return _run_tracking(args, mode="record")


def cmd_train(args: argparse.Namespace) -> int:
    from gesture_pilot.classifier import ClassifierError, GestureClassifier, format_confusion
    from gesture_pilot.dataset import DatasetError, load_dataset

    try:
        dataset = load_dataset(args.dataset)
        print(f"{len(dataset)} samples: " + ", ".join(f"{k} {v}" for k, v in dataset.counts().items()))
        classifier, report = GestureClassifier.train(
            dataset, algorithm=args.algorithm, test_size=args.test_size, seed=args.seed
        )
    except (DatasetError, ClassifierError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # e.g. a test split too small for the number of classes
        print(f"error: cannot split the dataset: {exc}", file=sys.stderr)
        return 1
    print(f"\n{report.algorithm}: trained on {report.train_samples}, tested on {report.test_samples}")
    print(report.report)
    print("confusion matrix (rows = true, columns = predicted):")
    print(format_confusion(report))
    classifier.save(args.output)
    print(f"\naccuracy {report.accuracy:.1%} - model saved to {args.output}")
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    from gesture_pilot.config import DEFAULT_CONFIG

    if args.path.exists() and not args.force:
        print(f"error: {args.path} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    print(f"wrote {args.path} - edit it, then run: gesture-pilot run --config {args.path}")
    return 0


COMMANDS = {
    "download-model": cmd_download_model,
    "preview": cmd_preview,
    "track": cmd_track,
    "recognize": cmd_recognize,
    "record": cmd_record,
    "train": cmd_train,
    "init-config": cmd_init_config,
    "run": cmd_run,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
