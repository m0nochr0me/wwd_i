import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.elevenlabs import (
    _balanced_voice_ids,
    _pcm16_to_float,
    _rms_normalize,
    _variants,
    generate_clips,
)


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


class _LabeledVoice:
    def __init__(self, vid, gender, age, accent):
        self.voice_id = vid
        self.labels = {"gender": gender, "age": age, "accent": accent}


def test_balanced_voice_ids_even_spread():
    # 3 male voices vs 9 female: round-robin across the two demographic buckets
    # should pull 3 of each for 6 picks, not just the 6 most-numerous (all female).
    voices = [_LabeledVoice(f"m{i}", "male", "young", "american") for i in range(3)]
    voices += [_LabeledVoice(f"f{i}", "female", "young", "british") for i in range(9)]
    sel = _balanced_voice_ids(voices, max_voices=6, seed=0)
    assert len(sel) == 6
    assert sum(s.startswith("m") for s in sel) == 3  # males not starved by the larger bucket
    assert _balanced_voice_ids(voices, 6, seed=0) == sel  # deterministic
    assert set(_balanced_voice_ids(voices, 99, seed=0)) == {v.voice_id for v in voices}  # caps at available


def test_generate_clips_writes_and_caches(tmp_path):
    calls = []

    def fake_synth(client, text, voice_id, *, model, stability, style, similarity):
        calls.append((text, voice_id, stability, style, similarity))
        return _sine_pcm()

    client = _FakeClient([f"v{i}" for i in range(10)])
    paths = generate_clips("hey computer", tmp_path, n_clips=8, client=client, synthesize=fake_synth, seed=0)
    assert len(paths) == 8 and len(calls) == 8
    for p in paths:
        wav, sr = sf.read(p)
        assert sr == SAMPLE_RATE and wav.ndim == 1

    again = generate_clips("hey computer", tmp_path, n_clips=8, client=client, synthesize=fake_synth, seed=0)
    assert again == paths and len(calls) == 8  # all cached -> no new synthesis


class _ApiError(Exception):
    """Mimics elevenlabs ApiError: carries status_code + body, duck-typed by the module."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"detail": {"message": message, "status": "voice_not_fine_tuned"}}


def test_generate_clips_skips_unusable_voice(tmp_path):
    bad = "v3"
    calls = []

    def fake_synth(client, text, voice_id, *, model, stability, style, similarity):
        calls.append(voice_id)
        if voice_id == bad:
            raise _ApiError(400, f"Voice '{voice_id}' is not fine-tuned and cannot be used.")
        return _sine_pcm()

    client = _FakeClient([f"v{i}" for i in range(5)])
    # n_clips == full grid (5 voices x 3 x 3 x 3 = 135) -> every voice is drawn, so the bad one is hit.
    paths = generate_clips("hey computer", tmp_path, n_clips=135, client=client, synthesize=fake_synth, seed=0)
    assert calls.count(bad) == 1  # attempted once, then blacklisted for the rest of the batch
    assert len(paths) == 4 * 27  # only the 4 good voices x 27 setting combos were written


def test_generate_clips_reraises_non_voice_error(tmp_path):
    def fake_synth(client, text, voice_id, *, model, stability, style, similarity):
        raise _ApiError(401, "invalid api key")  # auth error -> must not be swallowed

    client = _FakeClient([f"v{i}" for i in range(3)])
    try:
        generate_clips("hey computer", tmp_path, n_clips=5, client=client, synthesize=fake_synth, seed=0)
    except _ApiError as e:
        assert e.status_code == 401
    else:
        raise AssertionError("expected the 401 to propagate")
