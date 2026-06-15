import numpy as np

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.augment import Augmenter, apply_rir, mix_at_snr, speed_perturb, synthetic_rir


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


def test_augmenter_deterministic_and_uses_pool():
    pool = [np.random.default_rng(2).standard_normal(SAMPLE_RATE).astype(np.float32)]
    clip = np.full(SAMPLE_RATE, 0.1, np.float32)
    out_a = Augmenter(pool, seed=7)(clip.copy())
    out_b = Augmenter(pool, seed=7)(clip.copy())
    assert np.allclose(out_a, out_b)  # seeded -> reproducible
    assert out_a.dtype == np.float32 and np.all(np.isfinite(out_a))

    noisy = Augmenter(pool, p_gain=0, p_speed=0, p_rir=0, p_clip=0, p_noise=1.0, seed=0)(clip.copy())
    assert not np.allclose(noisy, clip)  # noise mixing changed the signal
