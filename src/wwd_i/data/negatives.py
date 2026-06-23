"""Negative material for head training (Phase 3).

Two kinds, per docs/architecture.md §7:

- **Hard negatives** — phrases acoustically near the wake phrase (its own
  sub-words plus a few generic commands/fillers), synthesized with the same
  TTS path as the positives. A starting set; extend per word as FA shows.
- **Background negatives** — random fixed-length crops of generic audio
  (speech/music/noise) from the background pool, labelled "not the wake word".
"""

from collections.abc import Iterable

import numpy as np

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.clips import fixed_length

# Generic confusables / common short utterances, independent of the wake phrase.
GENERIC_CONFUSABLES = (
    "hey",
    "okay",
    "hello there",
    "what time is it",
    "never mind",
    "play some music",
    "turn it off",
    "thank you",
    "no thanks",
    "one moment",
)


def hard_negative_phrases(wake_phrase: str, extra: Iterable[str] = ()) -> list[str]:
    """Near-phrase texts to synthesize as hard negatives for ``wake_phrase``.

    Its individual words (sub-phrases — a frequent false-trigger source), the
    generic confusables, and any ``extra`` phrases (e.g. LLM-generated acoustic
    confusables from :mod:`wwd_i.data.confusables`), minus anything equal to the
    full wake phrase. ``extra`` is lower-cased and de-duplicated with the rest.
    """
    wake = wake_phrase.strip().lower()
    words = wake.split()
    candidates = set(GENERIC_CONFUSABLES)
    candidates.update(s for e in extra if (s := e.strip().lower()))
    if len(words) > 1:
        candidates.update(words)  # each word alone
    candidates.discard(wake)
    candidates.discard("")
    return sorted(candidates)


def sample_background_clips(
    pool: list[np.ndarray],
    n: int,
    *,
    length: int = SAMPLE_RATE,
    seed: int = 0,
) -> list[np.ndarray]:
    """Draw ``n`` random ``length``-sample crops from the background pool."""
    if not pool:
        raise RuntimeError("empty background pool")
    rng = np.random.default_rng(seed)
    clips: list[np.ndarray] = []
    for _ in range(n):
        src = pool[int(rng.integers(len(pool)))]
        if len(src) > length:
            start = int(rng.integers(0, len(src) - length + 1))
            src = src[start : start + length]
        clips.append(fixed_length(src, length))
    return clips
