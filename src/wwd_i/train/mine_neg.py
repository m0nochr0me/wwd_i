"""Mine hard negatives: stream media through the engine and harvest the windows a
head false-fires on, as label-0 embeddings for the next retrain (Phase 4).

The recurring media false-accepts (conversational speech, TV/YT) come from held-out
audio the head never trained on; adding *more random* background negatives plateaus.
This targets the exact windows that fire: it streams continuous media through the real
``WakeWordEngine`` at a low ``--mine-thr`` and snapshots the trailing embedding window
(``engine.last_emb_window``, shape ``[HEAD_CONTEXT_HOPS, D]``) at every firing hop. The
output ``[N, W, D]`` cache is byte-compatible with ``preprocess_bg``'s ``bg_neg.npy`` —
pass both to ``train_head --bg-neg-emb`` (it concatenates).

Bootstrap loop: mine with the current head, retrain, re-mine with the new head on FRESH
media, repeat. HARD GUARD (yours to enforce): ``--media`` must be DISJOINT from ``--calib-bg``
and ``--rhythm-impostors``, else the gate becomes test-on-train. Torch-free (runtime only).

Run: ``uv run python -m wwd_i.train.mine_neg --head artifacts/<word>_head.onnx --media DIR --out data/mined_neg.npy``
"""

import argparse
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
from wwd_i.config import FRAME_SAMPLES
from wwd_i.data.backgrounds import find_audio
from wwd_i.runtime.engine import HEAD_CONTEXT_HOPS, WakeWordEngine


def mine_file(engine: WakeWordEngine, wav: np.ndarray, *, cap: int) -> list[np.ndarray]:
    """Stream one waveform hop-by-hop; return the trailing embedding windows (each
    ``[HEAD_CONTEXT_HOPS, D]``) at hops where the head fired (>= the engine's threshold), up to ``cap``.

    One ``FRAME_SAMPLES`` chunk advances exactly one 80 ms hop, so ``last_emb_window`` and the
    returned detections line up — streaming the whole file keeps the AGC warm (the real
    deployment condition), unlike scoring isolated crops.
    """
    engine.reset()
    out: list[np.ndarray] = []
    for start in range(0, len(wav), FRAME_SAMPLES):
        dets = engine.push(wav[start : start + FRAME_SAMPLES])
        win = engine.last_emb_window
        if dets and win is not None and win.shape[0] == HEAD_CONTEXT_HOPS:
            out.append(win.copy())
            if len(out) >= cap:
                break
    return out


def mine(args: argparse.Namespace) -> np.ndarray:
    paths = find_audio(args.media)
    if not paths:
        raise RuntimeError(f"no audio files under {args.media}")
    # threshold = mine_thr, refractory 0 => push() fires at every hop >= mine_thr (each a hard-negative window).
    engine = WakeWordEngine(args.head, threshold=args.mine_thr, refractory_s=0.0, backbone_path=args.backbone)

    mined: list[np.ndarray] = []
    scanned = skipped = 0
    print(f"mining up to {args.max} hard negatives from {len(paths)} files (thr {args.mine_thr})...", flush=True)
    for p in paths:
        if len(mined) >= args.max:
            break
        try:
            wav = load_wav(p)
        except Exception as e:  # noqa: BLE001 — one bad file must not kill a long mining run
            print(f"skipping unreadable {p}: {e}")
            skipped += 1
            continue
        mined.extend(mine_file(engine, wav, cap=args.max_per_file))
        scanned += 1
        if scanned % 50 == 0:
            print(f"  {scanned}/{len(paths)} files | {len(mined)} mined", flush=True)

    if not mined:
        raise RuntimeError(f"no windows scored >= --mine-thr {args.mine_thr} (lower it, or the head is already clean)")
    emb = np.stack(mined[: args.max]).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    print(f"done | {len(emb)} hard negatives {emb.shape} -> {out} | {scanned} files scanned, {skipped} skipped")
    return emb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mine hard-negative embeddings from media a head false-fires on (Phase 4).")
    p.add_argument("--head", required=True, help="head onnx whose false-fires to mine")
    p.add_argument("--media", nargs="+", required=True, help="dirs of continuous media (keep DISJOINT from --calib-bg)")
    p.add_argument("--out", default="data/mined_neg.npy", help="output [N, W, D] hard-negative cache")
    p.add_argument(
        "--mine-thr",
        type=float,
        default=0.3,
        help="snapshot windows scoring >= this (below the calibrated fire threshold)",
    )
    p.add_argument("--max", type=int, default=20000, help="stop after this many mined windows")
    p.add_argument("--max-per-file", type=int, default=200, help="cap per file so one track can't dominate the cache")
    p.add_argument("--backbone", default="", help="backbone onnx (default: packaged)")
    return p


def main() -> None:
    mine(build_parser().parse_args())


if __name__ == "__main__":
    main()
