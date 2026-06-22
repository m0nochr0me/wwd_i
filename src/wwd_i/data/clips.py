"""Shared helpers for word-folder audio datasets (Phase 2).

Both the Speech Commands bootstrap and the MSWC scale-up materialize clips as
``{root}/.../{word}/{clip}.wav`` and train from that layout. This module indexes
such a directory, loads clips through the inference runtime's loader (soundfile),
and builds the train / held-out episodic samplers used by metric learning.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
from wwd_i.config import SAMPLE_RATE
from wwd_i.data.episodic import EpisodicSampler


def fixed_length(wav: np.ndarray, length: int = SAMPLE_RATE) -> np.ndarray:
    """Crop or zero-pad a 1-D waveform to exactly ``length`` samples."""
    if len(wav) >= length:
        return wav[:length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[: len(wav)] = wav
    return out


def index_clips(root: str | Path) -> dict[str, list[Path]]:
    """Map word -> extracted ``.wav`` paths by parent folder, skipping noise."""
    index: dict[str, list[Path]] = {}
    for wav in sorted(Path(root).rglob("*.wav")):
        label = wav.parent.name
        if label == "_background_noise_":
            continue
        index.setdefault(label, []).append(wav)
    return index


def load_clip(label: str, path: Path) -> np.ndarray:
    """Load one clip to a fixed-length 16 kHz mono float32 waveform."""
    return fixed_length(load_wav(path), SAMPLE_RATE)


def split_samplers(
    index: dict[str, list[Path]],
    *,
    held_out_words: list[str] | tuple[str, ...],
    limit: int | None = None,
    seed: int = 0,
    augment: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Split a word index into (train_sampler, held_out_sampler).

    ``held_out_words`` are reserved for the few-shot probe and never trained on;
    ``limit`` caps clips kept per word (for fast runs). ``augment`` (if given) is
    applied to **training** clips only — the probe must stay clean — and its
    length-changing output (speed-perturb) is re-fixed to one clip length.
    """
    if limit is not None:
        index = {word: paths[:limit] for word, paths in index.items()}
    held_set = set(held_out_words)
    held = {word: index[word] for word in held_out_words if word in index}
    train = {word: paths for word, paths in index.items() if word not in held_set}
    if not train or not held:
        raise RuntimeError(f"empty split: {len(train)} train words, {len(held)} held-out words")

    train_load = load_clip
    if augment is not None:

        def train_load(label: str, path: Path) -> np.ndarray:
            return fixed_length(augment(load_clip(label, path)), SAMPLE_RATE)

    return (
        EpisodicSampler(train, train_load, seed=seed),
        EpisodicSampler(held, load_clip, seed=seed + 1),
    )
