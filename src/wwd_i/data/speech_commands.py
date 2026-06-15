"""Speech Commands v2 episodic dataset (Phase-2 bootstrap).

Wraps torchaudio's SPEECHCOMMANDS into a per-class index, splits the 35 words into
a training pool and a held-out (unseen) pool, and exposes an EpisodicSampler for
each. The held-out pool feeds the Phase-2 few-shot probe; the training pool feeds
prototypical training. This is the small-vocabulary bootstrap used to stand up and
verify the pipeline before the full MSWC run on Colab.

torchaudio is a training-only dep (the `train` group); imported lazily so a core
install never needs it. See docs/implementation-plan.md Phase 2.
"""

from pathlib import Path

import numpy as np

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.episodic import EpisodicSampler

# Ten words held out of training; the probe measures transfer to exactly these.
HELD_OUT_WORDS: tuple[str, ...] = (
    "backward",
    "bed",
    "bird",
    "cat",
    "dog",
    "eight",
    "five",
    "follow",
    "forward",
    "four",
)


def _fixed_length(wav: np.ndarray, length: int) -> np.ndarray:
    """Crop or zero-pad a 1-D waveform to exactly ``length`` samples."""
    if len(wav) >= length:
        return wav[:length].astype(np.float32)
    out = np.zeros(length, dtype=np.float32)
    out[: len(wav)] = wav
    return out


def speech_commands_samplers(
    root: str | Path, *, limit: int | None = None, seed: int = 0
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Return (train_sampler, held_out_sampler) over Speech Commands v2.

    Downloads the dataset under ``root`` on first use (~2.3 GB). ``limit`` caps the
    number of clips kept per word (for a fast local smoke run); ``None`` keeps all.
    """
    import torchaudio

    ds = torchaudio.datasets.SPEECHCOMMANDS(root=str(root), download=True)
    index: dict[str, list[int]] = {}
    for i in range(len(ds)):
        label = ds.get_metadata(i)[2]  # (relpath, sample_rate, label, speaker_id, utterance)
        index.setdefault(label, []).append(i)
    if limit is not None:
        index = {label: items[:limit] for label, items in index.items()}

    def load(label: str, item: int) -> np.ndarray:
        wav = ds[item][0].squeeze(0).numpy()  # [1, T] float32 @ 16 kHz -> [T]
        return _fixed_length(wav, SAMPLE_RATE)

    held = {w: index[w] for w in HELD_OUT_WORDS if w in index}
    train = {w: items for w, items in index.items() if w not in HELD_OUT_WORDS and w != "_background_noise_"}
    return (
        EpisodicSampler(train, load, seed=seed),
        EpisodicSampler(held, load, seed=seed + 1),
    )
