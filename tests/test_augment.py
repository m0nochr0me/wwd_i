import numpy as np
import pytest

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.augment import Augmenter, apply_rir, mix_at_snr, pitch_shift, speed_perturb, synthetic_rir


def _dominant_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x))
    spec[: int(50 * len(x) / SAMPLE_RATE)] = 0.0  # ignore DC / sub-50 Hz
    return float(np.argmax(spec) * SAMPLE_RATE / len(x))


def test_mix_at_snr_hits_target():
    rng = np.random.default_rng(0)
    clip = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    noise = rng.standard_normal(SAMPLE_RATE).astype(np.float32)
    out = mix_at_snr(clip, noise, 10.0, rng)
    added = out - clip  # the scaled noise component
    snr = 20 * np.log10(np.sqrt(np.mean(clip**2)) / np.sqrt(np.mean(added**2)))
    assert abs(snr - 10.0) < 0.5


def test_rir_and_apply_preserve_length():
    rng = np.random.default_rng(1)
    rir = synthetic_rir(rng)
    assert rir.ndim == 1 and np.isclose(np.max(np.abs(rir)), 1.0)
    assert len(apply_rir(rng.standard_normal(SAMPLE_RATE).astype(np.float32), rir)) == SAMPLE_RATE


def test_speed_perturb_changes_length():
    clip = np.zeros(SAMPLE_RATE, np.float32)
    assert len(speed_perturb(clip, 0.9)) > SAMPLE_RATE  # slower -> longer
    assert len(speed_perturb(clip, 1.1)) < SAMPLE_RATE  # faster -> shorter


def test_pitch_shift_no_op_when_zero():
    clip = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    out = pitch_shift(clip, 0.0)
    assert out.dtype == np.float32 and np.array_equal(out, clip)


@pytest.mark.parametrize("semitones", [-5.0, -2.5, 2.5, 5.0, 7.0])
def test_pitch_shift_preserves_length_and_shifts_pitch(semitones):
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE  # 1 s @ 16 kHz
    clip = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    out = pitch_shift(clip, semitones)
    assert out.dtype == np.float32 and len(out) == len(clip)  # duration preserved (decoupled from tempo)
    assert np.all(np.isfinite(out))
    expected = 220.0 * 2.0 ** (semitones / 12.0)
    assert abs(_dominant_hz(out) - expected) / expected < 0.05  # pitch moved by the right ratio


def test_pitch_shift_deterministic():
    clip = np.sin(2 * np.pi * 200 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    assert np.array_equal(pitch_shift(clip, 3.0), pitch_shift(clip, 3.0))


def test_augmenter_deterministic_and_uses_pool():
    pool = [np.random.default_rng(2).standard_normal(SAMPLE_RATE).astype(np.float32)]
    clip = np.full(SAMPLE_RATE, 0.1, np.float32)
    out_a = Augmenter(pool, seed=7)(clip.copy())
    out_b = Augmenter(pool, seed=7)(clip.copy())
    assert np.allclose(out_a, out_b)  # seeded -> reproducible
    assert out_a.dtype == np.float32 and np.all(np.isfinite(out_a))

    noisy = Augmenter(pool, p_gain=0, p_speed=0, p_rir=0, p_clip=0, p_noise=1.0, seed=0)(clip.copy())
    assert not np.allclose(noisy, clip)  # noise mixing changed the signal
