"""Label-free int8 regression guard: float vs quantized P(wake) (Phase 6).

Streams the same audio through a float engine and a quantized one and reports how far
the per-hop ``P(wake)`` sequence moved. Needs no labels and runs anywhere, mirroring the
Phase-5 streamed==batch parity philosophy: if the probability sequence barely moves the
calibrated threshold still holds; a large shift near a high threshold (heads calibrate at
0.99+) can flip detections and demands a full FA/FR re-gate on real data
(``train_head --backbone backbone.int8.onnx``).

This is the cheap, always-available half of the Phase-6 "accuracy regression guard"; the
authoritative half is the FA/FR re-gate, which needs the per-word eval set.

Run: ``uv run python -m wwd_i.export.compare <head.onnx> --int8-backbone backbone.int8.onnx --wavs <dir>``
"""

import argparse
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
from wwd_i.config import FRAME_SAMPLES
from wwd_i.runtime.engine import WakeWordEngine


def _pwake(head: str | Path, audio: np.ndarray, *, backbone_path: str = "") -> np.ndarray:
    """Per-hop ``P(wake)`` over ``audio`` (threshold 0 + no refractory => every hop fires,
    so each ``Detection.score`` is that hop's probability)."""
    eng = WakeWordEngine(head, threshold=0.0, refractory_s=0.0, backbone_path=backbone_path)
    return np.array([d.score for d in eng.push(np.asarray(audio, dtype=np.float32))], dtype=np.float64)


def compare(
    head: str | Path,
    audio: np.ndarray,
    *,
    int8_backbone: str | Path | None = None,
    int8_head: str | Path | None = None,
) -> dict:
    """Divergence of int8 vs float per-hop ``P(wake)`` on the same audio.

    Override the backbone, the head, or both. Returns max/RMS absolute probability
    difference and the largest float-vs-int8 probability pair (the worst hop), so a
    shift sitting right on a calibrated threshold is visible.
    """
    base = _pwake(head, audio)
    other = _pwake(int8_head or head, audio, backbone_path=str(int8_backbone) if int8_backbone else "")
    n = min(len(base), len(other))
    if n == 0:
        raise RuntimeError("no hops scored (audio shorter than one backbone window)")
    a, b = base[:n], other[:n]
    diff = np.abs(a - b)
    worst = int(diff.argmax())
    return {
        "hops": n,
        "max_abs": float(diff.max()),
        "rms": float(np.sqrt((diff**2).mean())),
        "worst_float": float(a[worst]),
        "worst_int8": float(b[worst]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare float vs int8 P(wake) (Phase-6 regression guard).")
    p.add_argument("head", help="float head .onnx")
    p.add_argument("--int8-backbone", help="quantized backbone to test against the packaged float one")
    p.add_argument("--int8-head", help="quantized head to test against the float head")
    p.add_argument(
        "--wavs",
        help="dir of wavs to stream. Use REAL audio incl. wake-word positives: the head saturates "
        "P(wake)->0 on non-speech, so the synthetic-noise default only smoke-tests the plumbing.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.wavs:
        audio = np.concatenate([load_wav(p) for p in sorted(Path(args.wavs).rglob("*.wav"))])
    else:
        audio = (np.random.default_rng(args.seed).standard_normal(FRAME_SAMPLES * 375) * 0.1).astype(np.float32)
    r = compare(args.head, audio, int8_backbone=args.int8_backbone, int8_head=args.int8_head)
    print(f"hops {r['hops']} | P(wake) max abs diff {r['max_abs']:.4f} | rms {r['rms']:.4f}")
    print(f"worst hop: float {r['worst_float']:.4f} vs int8 {r['worst_int8']:.4f}")


if __name__ == "__main__":
    main()
