"""Train a per-word wake head on frozen embeddings (Phase 4 driver).

Pipeline: ElevenLabs positives (+ hard negatives) and background negatives
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

import numpy as np
import onnxruntime as ort
import torch

from wwd_i.audio.io import load_wav
from wwd_i.config import MEL_FRAMES_PER_FRAME, SAMPLE_RATE
from wwd_i.data.augment import Augmenter
from wwd_i.data.backgrounds import load_background_pool
from wwd_i.data.clips import fixed_length
from wwd_i.data.negatives import sample_background_clips
from wwd_i.features.melspec import compute_logmel
from wwd_i.models.head import HeadConfig, WakeHead, export_head_onnx

WINDOW = 76  # mel frames fed to the backbone per hop (§3 rolling buffer ≈ 760 ms)
CLIP_SECONDS = 1.5


def _default_backbone() -> str:
    from importlib.resources import files

    return str(files("wwd_i.models") / "backbone.onnx")


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
        emb = session.run(None, {name: np.concatenate(seqs, axis=0)})[0]  # [len(chunk)*W, D]
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


def build_dataset(args: argparse.Namespace, aug: Augmenter) -> tuple[list[np.ndarray], np.ndarray]:
    length = int(CLIP_SECONDS * SAMPLE_RATE)
    pos_paths = sorted(Path(args.positives).rglob("*.wav"))
    if not pos_paths:
        raise RuntimeError(f"no positive wavs under {args.positives}")
    pos = _augmented(pos_paths, aug, args.n_aug, length)

    neg: list[np.ndarray] = []
    if args.hard_neg:
        neg += _augmented(sorted(Path(args.hard_neg).rglob("*.wav")), aug, args.n_aug, length)
    if aug.pool:
        bg = sample_background_clips(aug.pool, args.n_bg_neg, length=length, seed=args.seed)
        neg += [fixed_length(aug(c), length) for c in bg]
    if not neg:
        raise RuntimeError("no negatives — pass --background (recommended) and/or --hard-neg")

    labels = np.array([1.0] * len(pos) + [0.0] * len(neg), dtype=np.float32)
    print(f"dataset: {len(pos)} positive + {len(neg)} negative clips (n_aug={args.n_aug})")
    return pos + neg, labels


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
    clips, labels = build_dataset(args, aug)

    session = ort.InferenceSession(args.backbone, providers=["CPUExecutionProvider"])
    x = torch.from_numpy(embed_clips(clips, session)).float()
    y = torch.from_numpy(labels)

    # train / val (checkpoint selection) / test (honest calibration + gate)
    idx = np.random.default_rng(args.seed).permutation(len(x))
    n_hold = max(2, int(0.15 * len(x)))
    test, val, tr = idx[:n_hold], idx[n_hold : 2 * n_hold], idx[2 * n_hold :]

    head = WakeHead(HeadConfig(hidden=args.hidden, dropout=args.dropout))
    _fit(head, x, y, tr, val, args)

    head.eval()
    with torch.no_grad():
        test_prob = torch.sigmoid(head.clip_logits(x[test])).numpy()
    result = calibrate(test_prob, y[test].numpy(), target_fa=args.target_fa, target_fr=args.target_fr)

    c = result["chosen"]
    print(f"\nDET (held-out test, {result['neg_hours']:.2f} h negatives):")
    for r in result["det"]:
        mark = "  <-- chosen" if r is c else ""
        print(f"  thr {r['threshold']:.3f} | FR {r['fr']:.3f} | FA {r['fa_per_hr']:.2f}/hr{mark}")
    verdict = "PASS" if c["passed"] else "FAIL"
    print(
        f"\n[gate] thr {c['threshold']:.3f} -> FA {c['fa_per_hr']:.2f}/hr, FR {c['fr']:.3f} "
        f"(targets FA<{args.target_fa}/hr, FR<{args.target_fr}) {verdict}"
    )

    export_head_onnx(head, args.out)
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
    p.add_argument("--positives", required=True, help="dir of wake-phrase wavs (ElevenLabs)")
    p.add_argument("--hard-neg", help="dir of near-phrase wavs (ElevenLabs hard negatives)")
    p.add_argument("--background", nargs="*", help="dirs of background audio (AudioSet/FMA/MSWC) for noise + negatives")
    p.add_argument("--backbone", default=_default_backbone(), help="frozen backbone.onnx")
    p.add_argument("--n-aug", type=int, default=5, help="augmented variants per TTS clip")
    p.add_argument("--n-bg-neg", type=int, default=4000, help="background negative clips (more -> sharper FA/hr)")
    p.add_argument("--max-bg", type=int, default=2000, help="background clips decoded into the pool")
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
