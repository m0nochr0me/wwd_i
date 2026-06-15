import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.elevenlabs import _pcm16_to_float, _rms_normalize, _variants, generate_clips


class _FakeClient:
    """Stands in for ElevenLabs: only ``voices.get_all().voices`` is used."""

    def __init__(self, ids):
        self._ids = ids
        self.voices = self

    def get_all(self):
        return type("V", (), {"voices": [type("Voice", (), {"voice_id": i}) for i in self._ids]})


def _sine_pcm() -> bytes:
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def test_pcm_decode_and_rms_normalize():
    wav = _pcm16_to_float(_sine_pcm())
    assert wav.dtype == np.float32 and len(wav) == SAMPLE_RATE
    norm = _rms_normalize(wav, target_dbfs=-20.0)
    assert abs(20 * np.log10(np.sqrt(np.mean(norm**2))) + 20.0) < 0.5
    assert np.max(np.abs(norm)) <= 0.99 + 1e-6


def test_variants_deterministic_and_count():
    ids = [f"v{i}" for i in range(5)]
    assert _variants(20, ids, seed=0) == _variants(20, ids, seed=0)
    assert len(_variants(20, ids, seed=0)) == 20
    assert _variants(20, ids, seed=1) != _variants(20, ids, seed=0)


def test_generate_clips_writes_and_caches(tmp_path):
    calls = []

    def fake_synth(client, text, voice_id, *, model, stability, style):
        calls.append((text, voice_id, stability, style))
        return _sine_pcm()

    client = _FakeClient([f"v{i}" for i in range(10)])
    paths = generate_clips("hey computer", tmp_path, n_clips=8, client=client, synthesize=fake_synth, seed=0)
    assert len(paths) == 8 and len(calls) == 8
    for p in paths:
        wav, sr = sf.read(p)
        assert sr == SAMPLE_RATE and wav.ndim == 1

    again = generate_clips("hey computer", tmp_path, n_clips=8, client=client, synthesize=fake_synth, seed=0)
    assert again == paths and len(calls) == 8  # all cached -> no new synthesis
