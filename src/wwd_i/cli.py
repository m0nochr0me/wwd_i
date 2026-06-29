"""Command-line entry point for the streaming harness.

Two modes:

* Phase-0 throughput harness (default): stream a file or the microphone through
  an ONNX model (a generated identity model unless ``--model`` is given) and
  report throughput.
* Wake-word detection (``--head <word>_head.onnx``, repeatable): wire mel ->
  backbone -> head(s) and print detection events with timestamps. Pass ``--head``
  more than once to run several words off the one shared backbone; each event is
  tagged with the word that fired.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from wwd_i.audio import file_frames, mic_frames
from wwd_i.config import FRAME_MS, SAMPLE_RATE
from wwd_i.export import build_identity_model
from wwd_i.runtime import WakeWordEngine, load_session, run_stream
from wwd_i.runtime.engine import HEAD_CONTEXT_HOPS, HEAD_TOPK

DEFAULT_MODEL = Path("artifacts/identity.onnx")


def _run_detect(args: argparse.Namespace) -> int:
    engine = WakeWordEngine(
        args.head,
        calibration=args.calibration,
        threshold=args.threshold,
        refractory_s=args.refractory,
        normalize=not args.no_normalize,
    )
    frames = mic_frames() if args.mic else file_frames(args.source)
    for head in engine.heads:
        print(f"detecting {head.word!r} (threshold {head.threshold:.3f}, refractory {head.refractory_s:.2f}s)")
    if args.mic:
        print("listening — Ctrl-C to stop")

    report_every = max(1, round(1000 / FRAME_MS))  # ~1 s of frames between --debug lines
    captured: list[np.ndarray] = []
    count, compute, total = 0, 0.0, 0
    win_scores = [0.0] * len(engine.heads)  # per-head running max top-k-mean P(wake) since the last --debug report
    win_max = [0.0] * len(engine.heads)  # per-head running max top-1 (max-over-time) P(wake) — dilution diagnostic
    win_rms = 0.0
    try:
        for frame in frames:
            t0 = perf_counter()
            dets = engine.push(frame)
            compute += perf_counter() - t0
            count += 1
            for d in dets:
                total += 1
                print(
                    f"  \N{HIGH VOLTAGE SIGN} {d.word!r} @ {d.time_s:6.2f}s  score={d.score:.3f}  "
                    f"hops={d.hops_above}/{HEAD_CONTEXT_HOPS}  hi-mel={d.high_mel_energy:+.2f}"
                )
            if args.save is not None:
                captured.append(np.asarray(frame, dtype=np.float32))
            if args.debug:
                win_rms = max(win_rms, float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))
                win_scores = [max(w, h.last_score) for w, h in zip(win_scores, engine.heads, strict=True)]
                win_max = [max(w, h.last_max_score) for w, h in zip(win_max, engine.heads, strict=True)]
                if count % report_every == 0:
                    dbfs = 20 * np.log10(win_rms + 1e-9)
                    scores = "  ".join(
                        f"{h.word!r}=top{HEAD_TOPK} {w:.3f} top1 {m:.3f}"
                        for w, m, h in zip(win_scores, win_max, engine.heads, strict=True)
                    )
                    print(
                        f"  [debug] input {dbfs:6.1f} dBFS  hi-mel {engine.last_high_mel_energy:+.2f}  "
                        f"max P(wake) {scores}"
                    )
                    win_scores = [0.0] * len(engine.heads)
                    win_max = [0.0] * len(engine.heads)
                    win_rms = 0.0
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("\nstopped")
    finally:
        if args.save is not None and captured:
            import soundfile as sf

            sf.write(args.save, np.concatenate(captured), SAMPLE_RATE)
            print(f"saved {len(captured) * FRAME_MS / 1000:.2f}s of captured audio to {args.save}")

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
    parser.add_argument(
        "--head",
        type=Path,
        action="append",
        help="per-word head ONNX; switches to wake-word detection. Repeat to run several words "
        "off the one shared backbone (each detection is tagged with its word).",
    )
    parser.add_argument(
        "--calibration", type=Path, help="head calibration json (default: <head>.json); single --head only"
    )
    parser.add_argument("--threshold", type=float, help="override the calibrated detection threshold")
    parser.add_argument("--refractory", type=float, help="override the debounce window (seconds)")
    parser.add_argument(
        "--no-normalize", action="store_true", help="disable input loudness normalization (AGC) — for A/B debugging"
    )
    parser.add_argument(
        "--debug", action="store_true", help="print input level (dBFS) and max P(wake) ~once/sec, even below threshold"
    )
    parser.add_argument("--save", type=Path, help="write captured audio to this wav on exit (for offline replay)")
    args = parser.parse_args(argv)

    if args.mic == bool(args.source):
        parser.error("provide either an audio file or --mic (not both)")
    if args.head and len(args.head) > 1 and args.calibration is not None:
        parser.error("--calibration applies to a single --head; with multiple heads each uses its sibling <head>.json")

    return _run_detect(args) if args.head else _run_harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
