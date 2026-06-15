"""Load audio to the canonical format and slice a signal into fixed-size frames."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from wwd_i.config import FRAME_SAMPLES, SAMPLE_RATE


def load_wav(path: str | Path) -> np.ndarray:
    """Read an audio file as mono float32 in [-1, 1], resampled to SAMPLE_RATE."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)  # downmix channels
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def to_frames(signal: np.ndarray, frame_samples: int = FRAME_SAMPLES, *, pad: bool = False) -> Iterator[np.ndarray]:
    """Yield non-overlapping frames of `frame_samples` samples.

    A trailing partial frame is dropped unless `pad` is set, in which case it is
    zero-padded to a full frame.
    """
    n = len(signal)
    full = n // frame_samples
    for i in range(full):
        yield signal[i * frame_samples : (i + 1) * frame_samples]
    rem = n - full * frame_samples
    if pad and rem:
        last = np.zeros(frame_samples, dtype=np.float32)
        last[:rem] = signal[full * frame_samples :]
        yield last
