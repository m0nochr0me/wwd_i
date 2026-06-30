"""On-the-fly waveform augmentation for head training (Phase 3).

Clean TTS positives are too easy; real audio is reverberant, noisy,
and recorded at varied gains. This applies, per call, a random subset of: gain,
speed-perturb (joint tempo+pitch, Kaldi-style), duration-preserving pitch-shift
(formant/timbre — an axis the frozen backbone is NOT invariant to), reverberation
via a synthetic room impulse response, additive background at a sampled SNR, and clipping.
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


def _stft(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    """Centered (zero-padded) STFT -> complex ``[n_freq, n_frames]``."""
    x = np.pad(x, n_fft // 2)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1).T


def _istft(stft: np.ndarray, hop: int, win: np.ndarray, length: int) -> np.ndarray:
    """Invert ``_stft`` via windowed overlap-add, cropped to ``length`` samples."""
    n_fft = 2 * (stft.shape[0] - 1)
    frames = np.fft.irfft(stft.T, n_fft, axis=1) * win
    out_len = n_fft + hop * (stft.shape[1] - 1)
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    win2 = win**2
    for i in range(stft.shape[1]):
        s = i * hop
        y[s : s + n_fft] += frames[i]
        wsum[s : s + n_fft] += win2
    y /= np.maximum(wsum, _EPS)
    return y[n_fft // 2 : n_fft // 2 + length]


def _phase_vocoder(stft: np.ndarray, rate: float, hop: int) -> np.ndarray:
    """Time-stretch a complex STFT by ``rate`` (>1 shorter, <1 longer), phase-coherent."""
    n_freq = stft.shape[0]
    n_fft = 2 * (n_freq - 1)
    phi_advance = 2 * np.pi * hop * np.arange(n_freq) / n_fft
    steps = np.arange(0, stft.shape[1], rate)
    stft = np.pad(stft, [(0, 0), (0, 2)])  # guard the i+1 read on the last step
    out = np.empty((n_freq, len(steps)), dtype=complex)
    phase = np.angle(stft[:, 0])
    for t, step in enumerate(steps):
        i = int(step)
        frac = step - i
        mag = (1.0 - frac) * np.abs(stft[:, i]) + frac * np.abs(stft[:, i + 1])
        out[:, t] = mag * np.exp(1j * phase)
        dphi = np.angle(stft[:, i + 1]) - np.angle(stft[:, i]) - phi_advance
        dphi -= 2 * np.pi * np.round(dphi / (2 * np.pi))  # wrap to (-pi, pi]
        phase = phase + phi_advance + dphi
    return out


def pitch_shift(clip: np.ndarray, semitones: float, *, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Shift pitch by ``semitones`` while preserving duration (decoupled from tempo).

    A phase-vocoder time-stretch by ``1 / 2**(semitones/12)`` followed by a resample by the
    inverse restores the original length at the new pitch — unlike ``speed_perturb``, which
    couples the two. numpy + soxr only (no librosa). ``semitones == 0`` is a no-op.
    """
    clip = np.ascontiguousarray(clip, dtype=np.float32)
    if not semitones:
        return clip
    shift = 2.0 ** (semitones / 12.0)
    win = np.hanning(n_fft).astype(np.float64)
    stft = _stft(clip.astype(np.float64), n_fft, hop, win)
    stretched = _istft(_phase_vocoder(stft, 1.0 / shift, hop), hop, win, int(round(len(clip) * shift)))
    resampled = soxr.resample(stretched, int(round(SAMPLE_RATE * shift)), SAMPLE_RATE)
    n = len(clip)
    out = resampled[:n] if len(resampled) >= n else np.pad(resampled, (0, n - len(resampled)))
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
        p_gain: float = 0.1,
        gain_db: tuple[float, float] = (-6.0, 6.0),
        p_speed: float = 0.5,
        speed: tuple[float, float] = (0.9, 1.1),
        p_pitch: float = 0.3,
        pitch_semitones: tuple[float, float] = (-2.0, 2.0),
        p_rir: float = 0.5,
        p_noise: float = 0.8,
        snr_db: tuple[float, float] = (0.0, 20.0),
        p_clip: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.pool = background_pool or []
        # p_gain low by default: the backbone z-score-normalizes every window, so a constant gain
        # is analytically a no-op (export asserts loudness-invariance) — gain carries ~no signal.
        self.p_gain, self.gain_db = p_gain, gain_db
        self.p_speed, self.speed = p_speed, speed
        self.p_pitch, self.pitch_semitones = p_pitch, pitch_semitones
        self.p_rir = p_rir
        self.p_noise, self.snr_db = p_noise, snr_db
        self.p_clip = p_clip
        self.rng = np.random.default_rng(seed)

    def __call__(self, clip: np.ndarray) -> np.ndarray:
        rng = self.rng
        out = np.ascontiguousarray(clip, dtype=np.float32)
        if rng.random() < self.p_speed:
            out = speed_perturb(out, rng.uniform(*self.speed))
        if rng.random() < self.p_pitch:
            out = pitch_shift(out, rng.uniform(*self.pitch_semitones))
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
