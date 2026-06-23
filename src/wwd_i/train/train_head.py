"""Train a per-word wake head on frozen embeddings (Phase 4 driver).

Pipeline: TTS positives (+ hard negatives) and background negatives
(speech/music/noise) → on-the-fly-style augmentation baked into K variants → the
**frozen** ``backbone.onnx`` turns each ~1.5 s clip into an embedding sequence →
a tiny streaming GRU head is trained with max-over-time BCE → threshold is
calibrated to the FA/FR target and the head is exported to ``<word>_head.onnx``.

Run: ``uv run --group train python -m wwd_i.train.train_head --help``
See docs/architecture.md §6-§7 and implementation-plan.md Phase 4.
"""

import argparse
import copy
import json
from pathlib import Path
from typing import cast

# isort: off
import numpy as np

# torch must be imported before onnxruntime: importing torch loads the CUDA runtime and
# cuDNN libraries the onnxruntime-gpu wheel links against (they ship inside torch's
# nvidia-* dependencies), so `import onnxruntime` resolves on a GPU runtime instead of
# failing with `ImportError: libcudart.so.<N>`. See the ORT CUDA-EP docs (import torch /
# preload_dlls). `# isort: off/on` stops ruff from re-alphabetising past this order.
import torch
import onnxruntime as ort

# isort: on

from wwd_i.audio.io import load_wav
from wwd_i.config import MEL_FRAMES_PER_FRAME, SAMPLE_RATE
from wwd_i.data.augment import Augmenter
from wwd_i.data.backgrounds import find_audio, load_background_pool
from wwd_i.data.clips import fixed_length
from wwd_i.data.negatives import sample_background_clips
from wwd_i.features.melspec import compute_logmel
from wwd_i.models.head import HeadConfig, WakeHead, export_head_onnx
from wwd_i.runtime.engine import WakeWordEngine

WINDOW = 76  # mel frames fed to the backbone per hop (§3 rolling buffer ≈ 760 ms)
CLIP_SECONDS = 1.5


def _default_backbone() -> str:
    from importlib.resources import files

    return str(files("wwd_i.models") / "backbone.onnx")


def _make_session(backbone: str) -> ort.InferenceSession:
    """ORT session for the frozen backbone, on CUDA when available else CPU.

    The heavy embedding work (``preprocess_bg`` and the positives/hard-negs here)
    runs on the GPU when a Colab GPU runtime has the ``onnxruntime-gpu`` wheel
    (the notebook installs it when a GPU is present); otherwise it falls back to
    CPU. The active providers are printed so a silent CPU fallback is visible. The
    shipped runtime (``runtime/harness.py``) intentionally stays CPU-only.
    """
    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    session = ort.InferenceSession(backbone, providers=providers)
    print(f"backbone ORT providers: {session.get_providers()}", flush=True)
    return session


def _windows(mel: np.ndarray) -> np.ndarray:
    """log-mel ``[T, n_mels]`` -> stacked backbone inputs ``[W, WINDOW, n_mels]``."""
    hop = MEL_FRAMES_PER_FRAME
    n = (len(mel) - WINDOW) // hop + 1
    if n < 1:
        raise ValueError(f"clip too short: {len(mel)} mel frames < window {WINDOW}")
    return np.stack([mel[i * hop : i * hop + WINDOW] for i in range(n)]).astype(np.float32)


def embed_clips(clips: list[np.ndarray], session: ort.InferenceSession, *, batch: int = 256) -> np.ndarray:
    """Frozen-backbone embedding sequences for fixed-length clips -> ``[N, W, D]``.

    Embedded in chunks of ``batch`` clips (one ORT call each) with progress: a
    single call over the whole set allocates multi-GB backbone activations and
    appears to hang.
    """
    name = session.get_inputs()[0].name
    out: list[np.ndarray] = []
    print(f"embedding {len(clips)} clips through the frozen backbone...", flush=True)
    for start in range(0, len(clips), batch):
        chunk = clips[start : start + batch]
        seqs = [_windows(compute_logmel(c)) for c in chunk]
        emb = cast(np.ndarray, session.run(None, {name: np.concatenate(seqs, axis=0)})[0])  # [len(chunk)*W, D]
        out.append(emb.reshape(len(chunk), seqs[0].shape[0], emb.shape[-1]))
        print(f"  {min(start + batch, len(clips))}/{len(clips)}", flush=True)
    return np.concatenate(out, axis=0)


def _augmented(paths: list[Path], aug: Augmenter, n_aug: int, length: int) -> list[np.ndarray]:
    """Each clip plus ``n_aug`` augmented variants, all fixed to ``length`` samples."""
    out: list[np.ndarray] = []
    for p in paths:
        clip = fixed_length(load_wav(p), length)
        out.append(clip)
        out.extend(fixed_length(aug(clip), length) for _ in range(n_aug))
    return out


def build_embeddings(
    args: argparse.Namespace, aug: Augmenter, session: ort.InferenceSession
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(embeddings [N, W, D], labels [N])`` for the head.

    Positives and hard negatives are augmented clips embedded through the frozen
    backbone here. Bulk background negatives come **pre-embedded** from
    ``--bg-neg-emb`` (see ``preprocess_bg``) when given — bounded RAM regardless of
    count — else are sampled and embedded from the in-RAM pool (legacy path).
    """
    length = int(CLIP_SECONDS * SAMPLE_RATE)
    pos_paths = sorted(Path(args.positives).rglob("*.wav"))
    if not pos_paths:
        raise RuntimeError(f"no positive wavs under {args.positives}")
    pos_emb = embed_clips(_augmented(pos_paths, aug, args.n_aug, length), session)

    neg_parts: list[np.ndarray] = []
    if args.hard_neg:
        hard = _augmented(sorted(Path(args.hard_neg).rglob("*.wav")), aug, args.n_aug, length)
        if hard:
            neg_parts.append(embed_clips(hard, session))
    if args.bg_neg_emb:
        cache = np.load(args.bg_neg_emb)
        if cache.shape[1:] != pos_emb.shape[1:]:
            raise RuntimeError(
                f"--bg-neg-emb {args.bg_neg_emb} has shape {cache.shape}, expected (*, {pos_emb.shape[1]}, "
                f"{pos_emb.shape[2]}) — rebuild it with preprocess_bg against the current backbone"
            )
        neg_parts.append(cache)
    elif aug.pool:
        bg = sample_background_clips(aug.pool, args.n_bg_neg, length=length, seed=args.seed)
        neg_parts.append(embed_clips([fixed_length(aug(c), length) for c in bg], session))
    if not neg_parts:
        raise RuntimeError("no negatives — pass --bg-neg-emb and/or --background and/or --hard-neg")

    neg_emb = np.concatenate(neg_parts, axis=0)
    x = np.concatenate([pos_emb, neg_emb], axis=0)
    y = np.array([1.0] * len(pos_emb) + [0.0] * len(neg_emb), dtype=np.float32)
    print(f"dataset: {len(pos_emb)} positive + {len(neg_emb)} negative clips (n_aug={args.n_aug})")
    return x, y


_THRESHOLDS = np.concatenate([np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99, 0.995, 0.999]])


def calibrate(prob: np.ndarray, y: np.ndarray, *, target_fa: float, target_fr: float) -> dict:
    """Sweep thresholds (toward 1.0); pick the lowest-FR point meeting FA<target."""
    pos, neg = prob[y == 1], prob[y == 0]
    neg_hours = len(neg) * CLIP_SECONDS / 3600.0
    det = []
    for thr in _THRESHOLDS:
        fr = float(np.mean(pos < thr)) if len(pos) else 1.0
        fa = int(np.sum(neg >= thr))
        det.append({"threshold": round(float(thr), 3), "fr": fr, "fa_per_hr": fa / neg_hours, "fa": fa})
    feasible = [r for r in det if r["fa_per_hr"] < target_fa]
    chosen = min(feasible, key=lambda r: r["fr"]) if feasible else max(det, key=lambda r: r["threshold"])
    chosen["passed"] = bool(chosen["fa_per_hr"] < target_fa and chosen["fr"] < target_fr)
    return {"chosen": chosen, "det": det, "neg_hours": neg_hours}


def _count_fires(times: np.ndarray, scores: np.ndarray, thr: float, refractory: float) -> int:
    """Refractory-debounced threshold crossings — mirrors ``WakeWordEngine.push`` fire logic."""
    last = float("-inf")
    fires = 0
    for t, s in zip(times, scores, strict=True):
        if s >= thr and (t - last) >= refractory:
            fires += 1
            last = float(t)
    return fires


def calibrate_stream(
    engine: WakeWordEngine,
    neg_paths: list[Path],
    pos_paths: list[Path],
    *,
    target_fa: float,
    target_fr: float,
    refractory: float,
) -> dict:
    """Calibrate by running the actual streaming engine, not isolated clips.

    The runtime scores a sliding window every 80 ms and debounces fires by
    ``refractory`` seconds; ``calibrate`` (one max-over-time score per 1.5 s clip)
    ignores that cadence and under-counts FA. Here we stream continuous negative
    recordings and the positive clips through ``engine`` once — it is built with
    threshold 0 and zero refractory so ``push`` returns *every* hop — then sweep
    thresholds offline applying the real refractory debounce. FA/hr is fires over the
    true negative audio duration; FR is the fraction of positive utterances that never
    fire. Same return shape as ``calibrate`` so the reporting/gate code is shared.
    """
    if not neg_paths:
        raise RuntimeError("calibrate_stream: no negative audio files")
    neg_streams: list[np.ndarray] = []  # per file: [hops, 2] = (time_s, score)
    neg_seconds = 0.0
    for p in neg_paths:
        wav = load_wav(p)
        neg_seconds += len(wav) / SAMPLE_RATE
        engine.reset()
        dets = engine.push(wav)
        neg_streams.append(np.array([[d.time_s, d.score] for d in dets], dtype=np.float64).reshape(-1, 2))

    pos_best: list[float] = []
    for p in pos_paths:
        engine.reset()
        dets = engine.push(load_wav(p))
        pos_best.append(max((d.score for d in dets), default=0.0))
    pos = np.array(pos_best, dtype=np.float64)
    neg_hours = neg_seconds / 3600.0

    det = []
    for thr in _THRESHOLDS:
        fr = float(np.mean(pos < thr)) if len(pos) else 1.0
        fa = sum(_count_fires(s[:, 0], s[:, 1], float(thr), refractory) for s in neg_streams)
        fa_per_hr = fa / neg_hours if neg_hours else 0.0
        det.append({"threshold": round(float(thr), 3), "fr": fr, "fa_per_hr": fa_per_hr, "fa": fa})
    feasible = [r for r in det if r["fa_per_hr"] < target_fa]
    chosen = min(feasible, key=lambda r: r["fr"]) if feasible else max(det, key=lambda r: r["threshold"])
    chosen["passed"] = bool(chosen["fa_per_hr"] < target_fa and chosen["fr"] < target_fr)
    return {"chosen": chosen, "det": det, "neg_hours": neg_hours}


def _fit(head: WakeHead, x: torch.Tensor, y: torch.Tensor, tr: np.ndarray, val: np.ndarray, args) -> None:
    """Train in place, keeping the best-val-loss weights (early-stopping-style).

    A tiny head on a memorizable set overfits fast (train acc -> ~1.0 while val
    degrades), so we regularize (AdamW weight decay + dropout) and restore the
    lowest-val-loss checkpoint rather than the last epoch.
    """
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossfn = torch.nn.BCEWithLogitsLoss()
    best_loss, best_state = float("inf"), copy.deepcopy(head.state_dict())
    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), args.batch_size):
            batch = perm[i : i + args.batch_size]
            loss = lossfn(head.clip_logits(x[batch]), y[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            vloss = lossfn(head.clip_logits(x[val]), y[val]).item()
        if vloss < best_loss:
            best_loss, best_state = vloss, copy.deepcopy(head.state_dict())
        if epoch % args.log_every == 0 or epoch == args.epochs:
            with torch.no_grad():
                acc = ((torch.sigmoid(head.clip_logits(x[tr])) >= 0.5).float() == y[tr]).float().mean().item()
            print(f"epoch {epoch:>4} | train loss {loss.item():.4f} | val loss {vloss:.4f} | train acc {acc:.3f}")
    head.load_state_dict(best_state)
    print(f"restored best-val checkpoint (val loss {best_loss:.4f})")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    pool = load_background_pool(args.background, max_clips=args.max_bg, seed=args.seed) if args.background else []
    aug = Augmenter(pool, seed=args.seed)

    session = _make_session(args.backbone)
    x_np, y_np = build_embeddings(args, aug, session)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(x_np).float().to(device)
    y = torch.from_numpy(y_np).to(device)

    # train / val (checkpoint selection) / test (honest calibration + gate)
    idx = np.random.default_rng(args.seed).permutation(len(x))
    n_hold = max(2, int(0.15 * len(x)))
    test, val, tr = idx[:n_hold], idx[n_hold : 2 * n_hold], idx[2 * n_hold :]

    head = WakeHead(HeadConfig(hidden=args.hidden, dropout=args.dropout)).to(device)
    _fit(head, x, y, tr, val, args)

    head.eval()
    export_head_onnx(head, args.out)  # export first: streaming calibration loads this head

    if args.calib_bg:
        neg_paths = find_audio(args.calib_bg)
        if not neg_paths:
            raise RuntimeError(f"--calib-bg has no audio files under {args.calib_bg}")
        pos_paths = sorted(Path(args.positives).rglob("*.wav"))
        # threshold 0 + zero refractory => push() returns every hop; calibrate_stream
        # re-thresholds offline with the real --refractory.
        engine = WakeWordEngine(args.out, threshold=0.0, refractory_s=0.0)
        result = calibrate_stream(
            engine, neg_paths, pos_paths, target_fa=args.target_fa, target_fr=args.target_fr, refractory=args.refractory
        )
        calib_label = f"streaming, {result['neg_hours']:.2f} h continuous negatives ({len(neg_paths)} files)"
    else:
        with torch.no_grad():
            test_prob = torch.sigmoid(head.clip_logits(x[test])).cpu().numpy()
        result = calibrate(test_prob, y[test].cpu().numpy(), target_fa=args.target_fa, target_fr=args.target_fr)
        calib_label = f"held-out test, {result['neg_hours']:.2f} h negatives"

    c = result["chosen"]
    print(f"\nDET ({calib_label}):")
    for r in result["det"]:
        mark = "  <-- chosen" if r is c else ""
        print(f"  thr {r['threshold']:.3f} | FR {r['fr']:.3f} | FA {r['fa_per_hr']:.2f}/hr{mark}")
    verdict = "PASS" if c["passed"] else "FAIL"
    print(
        f"\n[gate] thr {c['threshold']:.3f} -> FA {c['fa_per_hr']:.2f}/hr, FR {c['fr']:.3f} "
        f"(targets FA<{args.target_fa}/hr, FR<{args.target_fr}) {verdict}"
    )
    meta = {
        "word": args.word,
        "threshold": c["threshold"],
        "refractory_seconds": args.refractory,
        "fa_per_hr": c["fa_per_hr"],
        "fr": c["fr"],
        "passed": c["passed"],
    }
    Path(args.threshold_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.threshold_out).write_text(json.dumps(meta, indent=2))
    print(f"done | head {args.out} | calibration {args.threshold_out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a per-word wake head on frozen embeddings (Phase 4).")
    p.add_argument("--word", default="wake", help="label for the calibration metadata")
    p.add_argument("--positives", required=True, help="dir of wake-phrase wavs (TTS)")
    p.add_argument("--hard-neg", help="dir of near-phrase wavs (TTS hard negatives)")
    p.add_argument("--background", nargs="*", help="dirs of background audio for augmentation noise (e.g. noise_pool)")
    p.add_argument(
        "--calib-bg",
        nargs="+",
        help="dirs of HELD-OUT continuous negative audio: calibrate FA/hr by streaming it through the real "
        "engine (sliding window + refractory) instead of the per-clip estimate. Hold out from training negatives.",
    )
    p.add_argument(
        "--bg-neg-emb", help="pre-embedded background negatives .npy from preprocess_bg (bypasses --n-bg-neg)"
    )
    p.add_argument("--backbone", default=_default_backbone(), help="frozen backbone.onnx")
    p.add_argument("--n-aug", type=int, default=5, help="augmented variants per TTS clip")
    p.add_argument(
        "--n-bg-neg",
        type=int,
        default=4000,
        help="background negatives sampled from the pool (legacy; ignored with --bg-neg-emb)",
    )
    p.add_argument("--max-bg", type=int, default=2000, help="background clips decoded into the augmentation pool")
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--epochs", type=int, default=60, help="best-val checkpoint is kept, so this is an upper bound")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW regularization (fights overfitting)")
    p.add_argument("--dropout", type=float, default=0.2, help="head dropout (fights overfitting)")
    p.add_argument("--target-fa", type=float, default=0.5, help="false accepts / hour gate")
    p.add_argument("--target-fr", type=float, default=0.05, help="false reject rate gate")
    p.add_argument("--refractory", type=float, default=1.0, help="debounce seconds (recorded for the runtime)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default="artifacts/head.onnx", help="exported per-word head")
    p.add_argument("--threshold-out", default="artifacts/head.json", help="calibration (threshold + metrics)")
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
