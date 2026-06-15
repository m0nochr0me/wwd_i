"""On-the-fly waveform augmentation for head training (Phase 3).

Clean ElevenLabs TTS positives are too easy; real audio is reverberant, noisy,
and recorded at varied gains. This applies, per call, a random subset of: gain,
speed-perturb (joint tempo+pitch, Kaldi-style), reverberation via a synthetic
room impulse response, additive background at a sampled SNR, and clipping.
numpy + soxr only — no librosa (no py3.14 wheel). See docs/architecture.md §7.
"""

import numpy as np
import soxr

from wwd_i.config import SAMPLE_RATE

_EPS = 1e-9


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + _EPS


def _fit(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Tile/crop ``noise`` to exactly ``length`` samples from a random offset."""
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    start = int(rng.integers(0, len(noise) - length + 1))
    return noise[start : start + length]


def mix_at_snr(clip: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add ``noise`` to ``clip`` scaled to the target signal-to-noise ratio (dB)."""
    noise = _fit(noise, len(clip), rng)
    gain = _rms(clip) / (_rms(noise) * 10.0 ** (snr_db / 20.0))
    return clip + noise * gain


def synthetic_rir(rng: np.random.Generator, *, rt60_range: tuple[float, float] = (0.1, 0.6)) -> np.ndarray:
    """A simple exponential-decay room impulse response (direct path + tail)."""
    rt60 = rng.uniform(*rt60_range)
    n = max(2, int(rt60 * SAMPLE_RATE))
    decay = np.exp(-6.9 * np.arange(n) / (rt60 * SAMPLE_RATE))  # -60 dB at rt60
    rir = rng.standard_normal(n).astype(np.float32) * decay
    rir[0] += 1.0  # direct path
    return rir / np.max(np.abs(rir))


def _fftconvolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(a, nfft) * np.fft.rfft(b, nfft), nfft)[:n]
    return out.astype(np.float32)


def apply_rir(clip: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a RIR, keeping the original length (causal head)."""
    return _fftconvolve(clip, rir)[: len(clip)]


def speed_perturb(clip: np.ndarray, factor: float) -> np.ndarray:
    """Resample by ``factor`` (changes tempo and pitch together)."""
    out = soxr.resample(clip, int(round(SAMPLE_RATE * factor)), SAMPLE_RATE)
    return np.ascontiguousarray(out, dtype=np.float32)


class Augmenter:
    """Randomly perturb a 16 kHz mono waveform to simulate real capture.

    Each component fires independently with its probability; output length may
    differ from input (speed-perturb), so callers normalize length downstream.
    Pass a ``background_pool`` (see ``backgrounds.py``) to enable noise mixing.
    """

    def __init__(
        self,
        background_pool: list[np.ndarray] | None = None,
        *,
        p_gain: float = 0.8,
        gain_db: tuple[float, float] = (-6.0, 6.0),
        p_speed: float = 0.5,
        speed: tuple[float, float] = (0.9, 1.1),
        p_rir: float = 0.5,
        p_noise: float = 0.8,
        snr_db: tuple[float, float] = (0.0, 20.0),
        p_clip: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.pool = background_pool or []
        self.p_gain, self.gain_db = p_gain, gain_db
        self.p_speed, self.speed = p_speed, speed
        self.p_rir = p_rir
        self.p_noise, self.snr_db = p_noise, snr_db
        self.p_clip = p_clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        rng = self.rng
        out = np.ascontiguousarray(clip, dtype=np.float32)
        if rng.random() < self.p_speed:
            out = speed_perturb(out, rng.uniform(*self.speed))
        if rng.random() < self.p_gain:
            out = out * 10.0 ** (rng.uniform(*self.gain_db) / 20.0)
        if rng.random() < self.p_rir:
            out = apply_rir(out, synthetic_rir(rng))
        if self.pool and rng.random() < self.p_noise:
            out = mix_at_snr(out, self.pool[int(rng.integers(len(self.pool)))], rng.uniform(*self.snr_db), rng)
        if rng.random() < self.p_clip:
            out = np.clip(out, -0.99, 0.99)
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 1.0:
            out = out / peak
        return np.ascontiguousarray(out, dtype=np.float32)
