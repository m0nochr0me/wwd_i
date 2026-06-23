import numpy as np
import pytest
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.tts import TtsBackend, Variant, generate_clips, rms_normalize


def _sine() -> np.ndarray:
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class _Boom(Exception):
    """A per-voice failure the backend classifies as skippable."""


class _FakeBackend(TtsBackend):
    """Cycles ``n_voices`` voices; raises ``_Boom`` (skippable) when synthesizing ``bad``."""

    def __init__(self, n_voices=10, bad=None):
        self.n_voices = n_voices
        self.bad = bad
        self.calls = []

    def variants(self, n_clips, *, seed):
        return [Variant(tag=f"v{i % self.n_voices}|{i}", voice=f"v{i % self.n_voices}") for i in range(n_clips)]

    def synthesize(self, phrase, variant):
        self.calls.append(variant.voice)
        if variant.voice == self.bad:
            raise _Boom(variant.voice)
        return _sine()

    def is_unusable_voice(self, exc):
        return isinstance(exc, _Boom)


def test_rms_normalize_hits_target_and_caps_peak():
    norm = rms_normalize(_sine() * 4.0, target_dbfs=-20.0)
    assert abs(20 * np.log10(np.sqrt(np.mean(norm**2))) + 20.0) < 0.5
    assert np.max(np.abs(norm)) <= 0.99 + 1e-6
    assert norm.dtype == np.float32


def test_generate_clips_writes_and_caches(tmp_path):
    backend = _FakeBackend()
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert len(paths) == 8 and len(backend.calls) == 8
    for p in paths:
        wav, sr = sf.read(p)
        assert sr == SAMPLE_RATE and wav.ndim == 1

    again = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert again == paths and len(backend.calls) == 8  # all cached -> no new synthesis


def test_generate_clips_skips_unusable_voice(tmp_path):
    # n_clips=20 over 10 voices -> each voice twice; the bad one is attempted once then blacklisted.
    backend = _FakeBackend(n_voices=10, bad="v3")
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=20, seed=0)
    assert backend.calls.count("v3") == 1
    assert len(paths) == 18  # 20 variants minus v3's two


def test_generate_clips_reraises_non_skippable_error(tmp_path):
    class _HardBackend(_FakeBackend):
        def synthesize(self, phrase, variant):
            raise ValueError("account-wide failure")

    with pytest.raises(ValueError):
        generate_clips("hey computer", tmp_path, _HardBackend(), n_clips=5, seed=0)
