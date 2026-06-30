"""Pluggable TTS backends for Phase-3 sample generation.

A *backend* turns a phrase into diverse spoken renderings; this module owns the
backend-independent machinery — caching, RMS-normalize to -20 dBFS, 16 kHz mono
wav layout, and per-voice skip — so adding a backend is just implementing
``TtsBackend`` (two methods) and dropping it into ``generate_clips``.

Backend choice is a *licensing* decision as much as a quality one: the training
positives/hard-negatives are used to train the per-word head, so the backend's
terms must permit using its output to train an ML model. A permissively-licensed
local TTS (see ``local_tts.py``) keeps the pipeline clean end-to-end; the
ElevenLabs backend is opt-in and carries a ToS restriction — see
``docs/data-licensing.md``.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.quality import clip_defects


@dataclass(frozen=True)
class Variant:
    """One rendering to synthesize.

    ``tag`` encodes the backend's diversity knobs (e.g. voice + prosody) and keys
    the cache together with the phrase. ``voice`` is the unit a backend skips when
    a provider rejects it (empty for backends without per-voice failures).
    ``params`` is opaque to ``generate_clips`` and handed back to ``synthesize``.
    """

    tag: str
    voice: str = ""
    params: dict = field(default_factory=dict)


class TtsBackend(ABC):
    """A TTS source for ``generate_clips``. Implement the two abstract methods."""

    @abstractmethod
    def variants(self, n_clips: int, *, seed: int) -> list[Variant]:
        """Enumerate up to ``n_clips`` diverse renderings (deterministic in ``seed``)."""

    @abstractmethod
    def synthesize(self, phrase: str, variant: Variant) -> np.ndarray:
        """Render ``phrase`` for ``variant`` as float32 mono @ ``SAMPLE_RATE`` (any loudness)."""

    def is_unusable_voice(self, exc: Exception) -> bool:
        """True if ``exc`` means *this voice* is unusable (skip it, keep the batch going).

        Default: never — any synth error aborts the run. ElevenLabs overrides this to
        skip voices the API rejects per-voice while still propagating account-wide errors.
        """
        return False


def rms_normalize(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Scale to ``target_dbfs`` RMS, then guard the peak. Heads are trained at -20 dBFS."""
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < 1e-6:
        return x.astype(np.float32)  # silence — leave as-is
    y = x * (10.0 ** (target_dbfs / 20.0) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def generate_clips(
    phrase: str,
    out_dir: str | Path,
    backend: TtsBackend,
    *,
    n_clips: int = 300,
    seed: int = 0,
) -> list[Path]:
    """Synthesize ``n_clips`` diverse renderings of ``phrase`` into ``out_dir``.

    Cached by (phrase, variant.tag): existing files are reused, so reruns and a
    bumped ``n_clips`` only synthesize the new variants. A voice the backend reports
    unusable (``is_unusable_voice``) is skipped for the rest of the batch, and a
    render with a structural defect (``quality.clip_defects`` — disfluent, truncated,
    late-onset, silent) is dropped before it reaches disk, so the returned count may
    be < ``n_clips``; other errors propagate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    skipped: set[str] = set()
    for variant in backend.variants(n_clips, seed=seed):
        if variant.voice and variant.voice in skipped:
            continue
        tag = hashlib.blake2b(f"{phrase}|{variant.tag}".encode(), digest_size=8).hexdigest()
        path = out_dir / f"{tag}.wav"
        if not path.exists():
            try:
                audio = backend.synthesize(phrase, variant)
            except Exception as exc:  # noqa: BLE001 - re-raised unless the backend says it's a skippable voice
                if not backend.is_unusable_voice(exc):
                    raise
                skipped.add(variant.voice)
                print(f"skip voice {variant.voice}: {exc}")
                continue
            audio = rms_normalize(audio)
            defects = clip_defects(audio)
            if defects:
                print(f"skip defective clip {variant.tag}: {','.join(defects)}")
                continue
            sf.write(path, audio, SAMPLE_RATE)
        if path not in written:
            written.append(path)
    return written
