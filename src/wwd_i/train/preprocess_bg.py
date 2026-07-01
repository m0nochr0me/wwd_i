"""Pre-embed background negatives once, so head training never decodes them (Phase 3/4).

``train_head`` used to decode the whole background pool (AudioSet/FMA) into RAM and
then sample ``--n-bg-neg`` crops on top — full-length FMA tracks (~30 s) make that
multi-GB, so a high ``--n-bg-neg`` (needed to resolve FA < 0.5/hr) OOM-kills the
runtime. This step does the heavy work **once**, streaming the corpus file-by-file
and embedding fixed-length crops through the **frozen** ``backbone.onnx`` into a
compact ``[N, W, D]`` cache (~4 KB/clip). ``train_head --bg-neg-emb`` then loads that
directly — memory is bounded regardless of how many negatives you ask for, and the
cache is word-independent so it is reused for every wake-word head.

A small ``noise_pool/`` of raw wavs is written alongside so positives can still get
additive-noise augmentation on a restored runtime where the raw corpus was discarded
(``train_head --background noise_pool``).

Run: ``uv run --group train python -m wwd_i.train.preprocess_bg --help``
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from wwd_i.audio.io import load_wav
from wwd_i.config import SAMPLE_RATE
from wwd_i.data.augment import Augmenter
from wwd_i.data.backgrounds import _quiet_stderr, find_audio, load_background_pool
from wwd_i.data.clips import fixed_length
from wwd_i.data.tts import rms_normalize
from wwd_i.train.train_head import CLIP_SECONDS, _default_backbone, _make_session, embed_clips


def _crops(wav: np.ndarray, n: int, length: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Up to ``n`` random ``length``-sample crops from one waveform (tiled if short)."""
    if len(wav) <= length:
        return [fixed_length(wav, length)]
    starts = rng.integers(0, len(wav) - length + 1, size=n)
    return [wav[s : s + length].astype(np.float32) for s in starts]


def preprocess(args: argparse.Namespace) -> np.ndarray:
    """Stream the background dirs into a ``[N, W, D]`` negative-embedding cache.

    Decodes one file at a time (corrupt/short skipped, like ``load_background_pool``),
    takes a few crops, and embeds in fixed-size buffers so only one buffer of raw audio
    is ever resident. Also writes the first ``--noise-pool-size`` crops as wavs for
    augmentation. Returns the saved embeddings.
    """
    paths = find_audio(args.background)
    if not paths:
        raise RuntimeError(f"no audio files under {args.background}")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(paths)

    session = _make_session(args.backbone)
    length = int(CLIP_SECONDS * SAMPLE_RATE)
    min_len = int(args.min_seconds * SAMPLE_RATE)

    # Optional: augment a fraction of crops before embedding (RIR + speed + pitch + noise; gain off —
    # every crop is RMS-normalized to -20 dBFS below regardless). The backbone is strongly loudness-
    # SENSITIVE and the engine AGC-normalizes to -20 dBFS at inference, so embedding crops at natural
    # loudness trains the head off the served manifold (persistent media FA that more negatives can't
    # fix). Normalizing before aug also makes the augmenter's clip/additive-noise-SNR act at -20 dBFS.
    # This changes the cache CONTENT (shape unchanged) — regen the shared bg_neg.npy when it changes.
    aug = None
    if args.aug_frac > 0:
        pool = load_background_pool(args.background, max_clips=args.aug_pool, seed=args.seed)
        aug = Augmenter(pool, p_gain=0.0, seed=args.seed)

    noise_dir = Path(args.noise_pool_dir)
    noise_dir.mkdir(parents=True, exist_ok=True)

    parts: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    embedded = n_noise = skipped = 0
    print(f"preprocessing {len(paths)} files -> up to {args.n_bg_neg} negative embeddings...", flush=True)
    for path in paths:
        if embedded >= args.n_bg_neg:
            break
        try:
            with _quiet_stderr():  # silence per-file decoder chatter from corrupt clips
                wav = load_wav(path)
        except Exception:
            skipped += 1
            continue
        if len(wav) < min_len or not np.all(np.isfinite(wav)):
            skipped += 1
            continue
        for crop in _crops(wav, args.crops_per_file, length, rng):
            if n_noise < args.noise_pool_size:
                sf.write(noise_dir / f"{n_noise:05d}.wav", crop, SAMPLE_RATE)  # noise pool stays raw (aug noise source)
                n_noise += 1
            if aug is not None and rng.random() < args.aug_frac:
                crop = fixed_length(
                    aug(rms_normalize(crop)), length
                )  # normalize before aug so clip/SNR act at -20 dBFS
            crop = rms_normalize(crop)  # every embedded crop at -20 dBFS: match the engine's inference AGC (see above)
            buf.append(crop)
            if len(buf) >= args.chunk:
                parts.append(embed_clips(buf, session))
                embedded += len(buf)
                buf = []
                if embedded >= args.n_bg_neg:
                    break
    if buf:
        parts.append(embed_clips(buf, session))
        embedded += len(buf)

    if not parts:
        raise RuntimeError(f"no decodable clips >= {args.min_seconds}s under {args.background}")
    emb = np.concatenate(parts, axis=0)[: args.n_bg_neg]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    if embedded < args.n_bg_neg:
        print(f"WARNING: corpus exhausted at {len(emb)} < requested {args.n_bg_neg} (add files or --crops-per-file)")
    aug_note = f" | aug-frac {args.aug_frac}" if aug is not None else ""
    print(
        f"done | {len(emb)} negatives {emb.shape} -> {out} | {n_noise} noise wavs -> {noise_dir} "
        f"| {skipped} files skipped{aug_note}"
    )
    return emb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pre-embed background negatives for head training (Phase 3/4).")
    p.add_argument("--background", nargs="+", required=True, help="dirs of background audio (AudioSet/FMA)")
    p.add_argument("--out", default="data/bg_neg.npy", help="output [N, W, D] negative-embedding cache")
    p.add_argument("--n-bg-neg", type=int, default=50000, help="target negative crops (more -> sharper FA/hr)")
    p.add_argument("--crops-per-file", type=int, default=8, help="random crops drawn per source file")
    p.add_argument("--chunk", type=int, default=2048, help="raw crops embedded per buffer (caps peak RAM)")
    p.add_argument("--noise-pool-dir", default="data/noise_pool", help="small raw-wav pool for positive augmentation")
    p.add_argument("--noise-pool-size", type=int, default=500, help="number of raw crops written to the noise pool")
    p.add_argument("--min-seconds", type=float, default=1.0, help="skip source files shorter than this")
    p.add_argument(
        "--aug-frac",
        type=float,
        default=0.0,
        help="fraction of crops to augment (RIR+speed+pitch+noise, gain off) before embedding; 0 (default) = "
        "clean cache, byte-compatible with prior runs. Changes cache CONTENT not shape — regen the cache when changed.",
    )
    p.add_argument(
        "--aug-pool",
        type=int,
        default=300,
        help="background clips decoded into the additive-noise pool when --aug-frac>0 (bounded RAM)",
    )
    p.add_argument("--backbone", default=_default_backbone(), help="frozen backbone.onnx")
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    preprocess(build_parser().parse_args())


if __name__ == "__main__":
    main()
