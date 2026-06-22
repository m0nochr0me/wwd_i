"""Speech Commands v2 episodic dataset (Phase-2 bootstrap).

Downloads/extracts Speech Commands v2 via torchaudio (download/extract only), then
indexes the extracted word folders and splits 10 held-out words from the rest. The
held-out pool feeds the few-shot probe; the rest feeds prototypical training. This
is the small-vocabulary bootstrap that validated the pipeline before the MSWC
scale-up (see ``mswc.py``).

torchaudio (the `train` group) is used only to download/extract; clips are indexed
by globbing and loaded with soundfile (see ``clips.py``), so the torchcodec/ffmpeg
backend is not required. See docs/implementation-plan.md Phase 2.
"""

from collections.abc import Callable
from pathlib import Path

from wwd_i.data.clips import index_clips, split_samplers
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


def speech_commands_samplers(
    root: str | Path,
    *,
    limit: int | None = None,
    seed: int = 0,
    augment: Callable | None = None,
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Return (train_sampler, held_out_sampler) over Speech Commands v2.

    Downloads and extracts the dataset under ``root`` on first use (~2.3 GB).
    ``limit`` caps the clips kept per word (for a fast local smoke run).
    ``augment`` (if given) perturbs training clips only; see ``split_samplers``.
    """
    import torchaudio

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)  # torchaudio writes the archive into root but won't create it
    torchaudio.datasets.SPEECHCOMMANDS(root=str(root), download=True)  # download + extract only

    index = index_clips(root)
    if not index:
        raise RuntimeError(f"no Speech Commands .wav files found under {root}")
    return split_samplers(index, held_out_words=HELD_OUT_WORDS, limit=limit, seed=seed, augment=augment)
