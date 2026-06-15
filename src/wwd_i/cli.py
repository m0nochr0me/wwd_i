"""Command-line entry point for the Phase 0 streaming harness.

Streams a file or the microphone through an ONNX model (a generated identity
model by default) and reports throughput.
"""

import argparse
from pathlib import Path

import numpy as np

from wwd_i.audio import file_frames, mic_frames
from wwd_i.export import build_identity_model
from wwd_i.runtime import load_session, run_stream

DEFAULT_MODEL = Path("artifacts/identity.onnx")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wwd-i", description="wwd_i streaming harness (Phase 0)")
    parser.add_argument("source", nargs="?", help="audio file to stream; omit when using --mic")
    parser.add_argument("--mic", action="store_true", help="stream from the default microphone")
    parser.add_argument("--model", type=Path, help="ONNX model to run (default: generated identity model)")
    args = parser.parse_args(argv)

    if args.mic == bool(args.source):
        parser.error("provide either an audio file or --mic (not both)")

    model_path = args.model
    if model_path is None:
        model_path = build_identity_model(DEFAULT_MODEL)
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


if __name__ == "__main__":
    raise SystemExit(main())
