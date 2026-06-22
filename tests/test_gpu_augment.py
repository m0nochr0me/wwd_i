import numpy as np
import pytest

pytest.importorskip("torchaudio")  # GPUAugmenter uses torchaudio.functional.resample

import torch  # noqa: E402

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.train.gpu_augment import GPUAugmenter  # noqa: E402


def _batch(b: int = 6) -> torch.Tensor:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    rows = [0.3 * np.sin(2 * np.pi * (200 + 70 * i) * t) for i in range(b)]
    return torch.from_numpy(np.stack(rows).astype(np.float32))


def _pool() -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    return [rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1 for _ in range(4)]


def test_shape_dtype_finite_and_bounded():
    audio = _batch()
    out = GPUAugmenter(_pool(), seed=0)(audio)
    assert out.shape == audio.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 1.0 + 1e-6  # peak renorm keeps it in range


def test_seeded_reproducible():
    audio = _batch()
    a = GPUAugmenter(_pool(), seed=7)(audio)
    b = GPUAugmenter(_pool(), seed=7)(audio)
    assert torch.allclose(a, b)


def test_all_disabled_is_identity():
    audio = _batch()  # peak 0.3 < 1, so renorm is a no-op
    out = GPUAugmenter(None, p_gain=0, p_speed=0, p_rir=0, p_noise=0, p_clip=0, seed=0)(audio)
    assert torch.allclose(out, audio)


def test_noise_only_changes_signal_and_keeps_length():
    audio = _batch()
    out = GPUAugmenter(_pool(), p_gain=0, p_speed=0, p_rir=0, p_clip=0, p_noise=1.0, seed=0)(audio)
    assert out.shape == audio.shape
    assert not torch.allclose(out, audio)


def test_speed_only_keeps_length():
    audio = _batch()
    out = GPUAugmenter(None, p_gain=0, p_speed=1.0, p_rir=0, p_noise=0, p_clip=0, seed=0)(audio)
    assert out.shape == audio.shape  # length re-fixed after resample
    assert not torch.allclose(out, audio)


def test_reverb_only_changes_signal_and_keeps_length():
    audio = _batch()
    out = GPUAugmenter(None, p_gain=0, p_speed=0, p_rir=1.0, rt60=(0.1, 0.2), p_noise=0, p_clip=0, seed=0)(audio)
    assert out.shape == audio.shape
    assert not torch.allclose(out, audio)
