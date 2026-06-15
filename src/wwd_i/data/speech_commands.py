"""Speech Commands v2 episodic dataset (Phase-2 bootstrap).

Wraps torchaudio's SPEECHCOMMANDS into a per-class index, splits the 35 words into
a training pool and a held-out (unseen) pool, and exposes an EpisodicSampler for
each. The held-out pool feeds the Phase-2 few-shot probe; the training pool feeds
prototypical training. This is the small-vocabulary bootstrap used to stand up and
verify the pipeline before the full MSWC run on Colab.

torchaudio (the `train` group) is used only to download and extract the corpus;
clips are then indexed by globbing and loaded with soundfile (the same path the
inference runtime uses), so the torchcodec/ffmpeg backend is not required. See
docs/implementation-plan.md Phase 2.
"""

from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav
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


def _index_clips(root: Path) -> dict[str, list[Path]]:
    """Map word -> extracted ``.wav`` paths by parent folder, skipping noise/splits."""
    index: dict[str, list[Path]] = {}
    for wav in sorted(root.rglob("*.wav")):
        label = wav.parent.name
        if label == "_background_noise_":
            continue
        index.setdefault(label, []).append(wav)
    return index


def speech_commands_samplers(
    root: str | Path, *, limit: int | None = None, seed: int = 0
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Return (train_sampler, held_out_sampler) over Speech Commands v2.

    Downloads and extracts the dataset under ``root`` on first use (~2.3 GB).
    ``limit`` caps the clips kept per word (for a fast local smoke run).
    """
    import torchaudio

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)  # torchaudio writes the archive into root but won't create it
    torchaudio.datasets.SPEECHCOMMANDS(root=str(root), download=True)  # download + extract only

    index = _index_clips(root)
    if not index:
        raise RuntimeError(f"no Speech Commands .wav files found under {root}")
    if limit is not None:
        index = {label: paths[:limit] for label, paths in index.items()}

    def load(label: str, item: Path) -> np.ndarray:
        return _fixed_length(load_wav(item), SAMPLE_RATE)

    held = {w: index[w] for w in HELD_OUT_WORDS if w in index}
    train = {w: paths for w, paths in index.items() if w not in HELD_OUT_WORDS}
    return (
        EpisodicSampler(train, load, seed=seed),
        EpisodicSampler(held, load, seed=seed + 1),
    )
