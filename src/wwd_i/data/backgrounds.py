"""Background audio pool for head-training augmentation (Phase 3).

Additive noise and music (for augmentation) and background negatives are drawn
from generic audio corpora — AudioSet (noise/general) and FMA (music). The
notebook downloads and extracts those; this module just decodes a capped,
shuffled sample of the audio files under one or more directories to the canonical
16 kHz mono float32, skipping anything unreadable. See docs/architecture.md §7.
"""

import os
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
from wwd_i.config import SAMPLE_RATE

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")


@contextmanager
def _quiet_stderr():
    """Silence C-level stderr (fd 2): corrupt clips make libmpg123/libsndfile print
    resync/header errors directly, which would otherwise flood the training log."""
    sys.stderr.flush()
    saved, devnull = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def find_audio(dirs: str | Path | Iterable[str | Path]) -> list[Path]:
    """Recursively list audio files under one or more directories (sorted)."""
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    found: list[Path] = []
    for d in dirs:
        for path in sorted(Path(d).rglob("*")):
            if path.suffix.lower() in AUDIO_EXTS:
                found.append(path)
    return found


def load_background_pool(
    dirs: str | Path | Iterable[str | Path],
    *,
    max_clips: int = 2000,
    min_seconds: float = 1.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Decode up to ``max_clips`` random background waveforms (16 kHz mono float32).

    Files are shuffled (seeded) and decoded until ``max_clips`` clips of at least
    ``min_seconds`` are collected. Corrupt, too-short, or non-finite clips are
    skipped (corpora like FMA ship a few broken mp3s); the decoder's own error
    chatter is suppressed and a skip count is reported instead. Returns the raw
    variable-length waveforms — the augmenter crops/tiles them as needed.
    """
    paths = find_audio(dirs)
    if not paths:
        raise RuntimeError(f"no audio files ({', '.join(AUDIO_EXTS)}) under {dirs}")
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    min_len = int(min_seconds * SAMPLE_RATE)
    pool: list[np.ndarray] = []
    skipped = 0
    with _quiet_stderr():
        for path in paths:
            if len(pool) >= max_clips:
                break
            try:
                wav = load_wav(path)
            except Exception:
                skipped += 1  # corrupt / unsupported codec
                continue
            if len(wav) >= min_len and np.all(np.isfinite(wav)):
                pool.append(wav)
            else:
                skipped += 1  # too short or garbage from a partial decode
    if not pool:
        raise RuntimeError(f"no decodable clips >= {min_seconds}s under {dirs}")
    if skipped:
        print(f"background pool: {len(pool)} clips ({skipped} corrupt/short skipped)", flush=True)
    return pool
