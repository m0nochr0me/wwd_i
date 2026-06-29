import numpy as np
import pytest

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.elevenlabs import (
    ElevenLabsBackend,
    _balanced_voice_ids,
    _is_unusable_voice,
    _pcm16_to_float,
    _variants,
)
from wwd_i.data.tts import Variant, generate_clips


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


def test_pcm_decode():
    wav = _pcm16_to_float(_sine_pcm())
    assert wav.dtype == np.float32 and len(wav) == SAMPLE_RATE
    assert np.max(np.abs(wav)) <= 1.0


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


def test_backend_variants_and_synthesize_decode():
    captured = []

    def fake_synth(client, text, voice_id, *, model, stability, style, similarity, speed):
        captured.append((text, voice_id, model, stability, style, similarity, speed))
        return _sine_pcm()

    backend = ElevenLabsBackend(client=_FakeClient([f"v{i}" for i in range(5)]), synthesize=fake_synth)
    variants = backend.variants(10, seed=0)
    assert len(variants) == 10 and all(isinstance(v, Variant) for v in variants)
    assert all(v.voice.startswith("v") for v in variants)

    wav = backend.synthesize("hey computer", variants[0])
    assert wav.dtype == np.float32 and len(wav) == SAMPLE_RATE
    assert captured[0][0] == "hey computer" and captured[0][1] == variants[0].voice


class _ApiError(Exception):
    """Mimics elevenlabs ApiError: carries status_code + body, duck-typed by the module."""

    def __init__(self, status_code, message, *, status):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"detail": {"message": message, "status": status}}


@pytest.mark.parametrize(
    "status, skippable",
    [
        ("voice_not_fine_tuned", True),
        ("voice_disabled", True),
        ("invalid_api_key", False),
    ],
)
def test_is_unusable_voice_classification(status, skippable):
    assert _is_unusable_voice(_ApiError(400, "x", status=status)) is skippable


def test_backend_skips_unusable_voice_through_generate_clips(tmp_path):
    bad = "v3"
    calls = []

    def fake_synth(client, text, voice_id, *, model, stability, style, similarity, speed):
        calls.append(voice_id)
        if voice_id == bad:
            raise _ApiError(400, "not fine-tuned", status="voice_not_fine_tuned")
        return _sine_pcm()

    backend = ElevenLabsBackend(client=_FakeClient([f"v{i}" for i in range(5)]), synthesize=fake_synth)
    # n_clips == full grid (5 voices x 3 x 3 x 3 x 3 = 405) -> every voice drawn, so the bad one is hit.
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=405, seed=0)
    assert calls.count(bad) == 1  # attempted once, then blacklisted for the rest of the batch
    assert len(paths) == 4 * 81  # only the 4 good voices x 81 setting combos were written


def test_backend_reraises_non_voice_error(tmp_path):
    def fake_synth(client, text, voice_id, *, model, stability, style, similarity, speed):
        raise _ApiError(401, "invalid api key", status="invalid_api_key")

    backend = ElevenLabsBackend(client=_FakeClient([f"v{i}" for i in range(3)]), synthesize=fake_synth)
    with pytest.raises(_ApiError) as e:
        generate_clips("hey computer", tmp_path, backend, n_clips=5, seed=0)
    assert e.value.status_code == 401
