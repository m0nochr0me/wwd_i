"""Command-line entry point for the streaming harness.

Two modes:

* Phase-0 throughput harness (default): stream a file or the microphone through
  an ONNX model (a generated identity model unless ``--model`` is given) and
  report throughput.
* Wake-word detection (``--head <word>_head.onnx``): wire mel -> backbone -> head
  and print detection events with timestamps.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from wwd_i.audio import file_frames, mic_frames
from wwd_i.config import FRAME_MS
from wwd_i.export import build_identity_model
from wwd_i.runtime import WakeWordEngine, load_session, run_stream

DEFAULT_MODEL = Path("artifacts/identity.onnx")


def _run_detect(args: argparse.Namespace) -> int:
    engine = WakeWordEngine(
        args.head,
        calibration=args.calibration,
        threshold=args.threshold,
        refractory_s=args.refractory,
    )
    frames = mic_frames() if args.mic else file_frames(args.source)
    print(
        f"detecting {engine.word!r} (threshold {engine.threshold:.3f}, "
        f"refractory {engine.refractory_s:.2f}s){' — Ctrl-C to stop' if args.mic else ''}"
    )

    count, compute, total = 0, 0.0, 0
    try:
        for frame in frames:
            t0 = perf_counter()
            dets = engine.push(frame)
            compute += perf_counter() - t0
            count += 1
            for d in dets:
                total += 1
                print(f"  \N{HIGH VOLTAGE SIGN} {engine.word!r} @ {d.time_s:6.2f}s  score={d.score:.3f}")
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nstopped")
        return 0

    audio_ms = count * FRAME_MS
    rtf = (compute * 1000.0 / audio_ms) if audio_ms else 0.0
    print(f"detections={total} frames={count} audio={audio_ms / 1000:.2f}s rtf={rtf:.3f}")
    return 0


def _run_harness(args: argparse.Namespace) -> int:
    model_path = args.model or build_identity_model(DEFAULT_MODEL)
    if args.model is None:
        print(f"using generated identity model at {model_path}")
    session = load_session(model_path)

    peak = 0.0

    def _track(_index: int, out: np.ndarray) -> None:
        nonlocal peak
        peak = max(peak, float(np.abs(out).max()))

    if args.mic:
        print("streaming from microphone — Ctrl-C to stop")
        frames = mic_frames()
    else:
        frames = file_frames(args.source)

    try:
        stats = run_stream(session, frames, on_output=_track)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nstopped")
        return 0

    print(
        f"frames={stats.frames} audio={stats.audio_ms / 1000:.2f}s "
        f"compute={stats.compute_ms:.1f}ms rtf={stats.real_time_factor:.3f} peak={peak:.3f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wwd-i", description="wwd_i streaming harness")
    parser.add_argument("source", nargs="?", help="audio file to stream; omit when using --mic")
    parser.add_argument("--mic", action="store_true", help="stream from the default microphone")
    parser.add_argument("--model", type=Path, help="ONNX model for the throughput harness (default: identity)")
    parser.add_argument("--head", type=Path, help="per-word head ONNX; switches to wake-word detection")
    parser.add_argument("--calibration", type=Path, help="head calibration json (default: <head>.json)")
    parser.add_argument("--threshold", type=float, help="override the calibrated detection threshold")
    parser.add_argument("--refractory", type=float, help="override the debounce window (seconds)")
    args = parser.parse_args(argv)

    if args.mic == bool(args.source):
        parser.error("provide either an audio file or --mic (not both)")

    return _run_detect(args) if args.head else _run_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
