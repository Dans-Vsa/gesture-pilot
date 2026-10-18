"""Command-line interface: ``gesture-pilot <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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
    return parser


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
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), last):
            print(f"error: could not write snapshot {args.snapshot}", file=sys.stderr)
            return 1
    height, width = last.shape[:2]
    print(
        f"{source.name}: {counter.frames} frames, {width}x{height}, "
        f"average {counter.average:.1f} FPS"
    )
    return 0


COMMANDS = {
    "download-model": cmd_download_model,
    "preview": cmd_preview,
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
