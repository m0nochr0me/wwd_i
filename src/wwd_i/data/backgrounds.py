"""Background audio pool for head-training augmentation (Phase 3).

Additive noise and music (for augmentation) and background negatives are drawn
from generic audio corpora — AudioSet (noise/general) and FMA (music). The
notebook downloads and extracts those; this module just decodes a capped,
shuffled sample of the audio files under one or more directories to the canonical
16 kHz mono float32, skipping anything unreadable. See docs/architecture.md §7.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
from wwd_i.config import SAMPLE_RATE

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")


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
    ``min_seconds`` are collected; unreadable files are skipped. Returns the raw
    variable-length waveforms — the augmenter crops/tiles them as needed.
    """
    paths = find_audio(dirs)
    if not paths:
        raise RuntimeError(f"no audio files ({', '.join(AUDIO_EXTS)}) under {dirs}")
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    min_len = int(min_seconds * SAMPLE_RATE)
    pool: list[np.ndarray] = []
    for path in paths:
        if len(pool) >= max_clips:
            break
        try:
            wav = load_wav(path)
        except Exception:
            continue  # corrupt / unsupported codec — skip
        if len(wav) >= min_len:
            pool.append(wav)
    if not pool:
        raise RuntimeError(f"no decodable clips >= {min_seconds}s under {dirs}")
    return pool
