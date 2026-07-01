"""Audit (and optionally prune) a word's TTS positives by streaming-engine score.

Glitched TTS renders (wrong phonemes — e.g. an "Axana" clip that comes out
"adarachana") and too-short clips get labelled positive but the head cannot fire
on them. Left in, they poison training (mislabeled positives) and inflate the
calibration FR (every correct rejection counts as a miss). This streams each
positive through the real engine at threshold 0 and reports the best P(wake),
hop count, and duration so the bad ones sort to the top; ``--prune`` moves them out.

Score with an existing head for the word (the bimodal split — glitches ~0.00 vs
good renders ~0.98 — separates them cleanly). It is a bootstrap: re-audit after
retraining on the cleaned set. Torch-free — uses only the runtime engine.

Run: ``uv run python -m wwd_i.train.audit_positives --head artifacts/<word>_head.onnx --positives DIR``
"""

import argparse
import shutil
from pathlib import Path

from wwd_i.audio.io import load_wav
from wwd_i.config import SAMPLE_RATE
from wwd_i.runtime.engine import WakeWordEngine


def score_positives(engine: WakeWordEngine, paths: list[Path]) -> list[tuple[Path, float, int, float]]:
    """``(path, best_score, n_hops, seconds)`` for each positive streamed once through the engine."""
    rows: list[tuple[Path, float, int, float]] = []
    for p in paths:
        try:
            wav = load_wav(p)
        except Exception as e:  # noqa: BLE001 — one unreadable file must not kill the audit
            print(f"skipping unreadable {p}: {e}")
            continue
        engine.reset()
        dets = engine.push(wav)
        best = max((d.score for d in dets), default=0.0)
        rows.append((p, best, len(dets), len(wav) / SAMPLE_RATE))
    return rows


def audit(args: argparse.Namespace) -> list[tuple[Path, float, int, float]]:
    paths = sorted(Path(args.positives).rglob("*.wav"))
    if not paths:
        raise RuntimeError(f"no positive wavs under {args.positives}")
    # threshold 0 => push() returns every hop, so best_score is the clip's peak P(wake).
    engine = WakeWordEngine(args.head, threshold=0.0, refractory_s=0.0, backbone_path=args.backbone)
    rows = sorted(score_positives(engine, paths), key=lambda r: r[1])  # worst first
    flagged = [r for r in rows if r[1] < args.min_score]

    if args.out_csv:
        lines = ["path,best_score,n_hops,seconds"]
        lines += [f"{p},{s:.4f},{h},{sec:.2f}" for p, s, h, sec in rows]
        Path(args.out_csv).write_text("\n".join(lines) + "\n")

    print(f"{len(rows)} positives scored | {len(flagged)} below --min-score {args.min_score}")
    for p, s, h, sec in flagged[: args.show]:
        print(f"  {s:.3f} | {h:>3} hops | {sec:.2f}s | {p}")
    if len(flagged) > args.show:
        print(f"  ... {len(flagged) - args.show} more (see --out-csv)")

    if args.prune and flagged:
        rej = (
            Path(args.reject_dir)
            if args.reject_dir
            else Path(args.positives).parent / f"{Path(args.positives).name}_rejects"
        )
        rej.mkdir(parents=True, exist_ok=True)
        for p, *_ in flagged:
            shutil.move(str(p), str(rej / p.name))
        print(f"moved {len(flagged)} flagged positives -> {rej}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit/prune a word's TTS positives by streaming-engine score (Phase 4).")
    p.add_argument("--head", required=True, help="trained head onnx for the word (scores the positives)")
    p.add_argument("--positives", required=True, help="dir of positive wavs to audit")
    p.add_argument("--min-score", type=float, default=0.3, help="flag positives whose best P(wake) is below this")
    p.add_argument("--backbone", default="", help="backbone onnx (default: packaged)")
    p.add_argument("--out-csv", help="write the full (path,best_score,n_hops,seconds) table, sorted worst-first")
    p.add_argument("--show", type=int, default=20, help="flagged rows to print")
    p.add_argument(
        "--prune", action="store_true", help="MOVE flagged positives out to a rejects dir (default: report only)"
    )
    p.add_argument("--reject-dir", help="where --prune moves flagged files (default: <positives>_rejects sibling)")
    return p


def main() -> None:
    audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
