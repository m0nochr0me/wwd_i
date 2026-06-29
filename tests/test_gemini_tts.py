import numpy as np
import pytest

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.gemini_tts import (
    GEMINI_SAMPLE_RATE,
    STYLES,
    VOICES,
    GeminiTtsBackend,
    _pcm_to_float16k,
    _prompt,
    _variants,
)
from wwd_i.data.tts import Variant, generate_clips


def _sine_pcm(seconds: float = 1.0, rate: int = GEMINI_SAMPLE_RATE) -> bytes:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    return (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def test_pcm_decode_resamples_to_16k():
    wav = _pcm_to_float16k(_sine_pcm(seconds=1.0))
    assert wav.dtype == np.float32
    assert abs(len(wav) - SAMPLE_RATE) <= 2  # 24 kHz -> 16 kHz, one second
    assert np.max(np.abs(wav)) <= 1.0


def test_prompt_neutral_vs_styled():
    assert _prompt("hey computer", "") == "hey computer"
    assert _prompt("hey computer", "Say cheerfully") == "Say cheerfully: hey computer"


def test_variants_deterministic_and_count():
    assert _variants(50, seed=0) == _variants(50, seed=0)
    assert len(_variants(50, seed=0)) == 50
    assert _variants(50, seed=1) != _variants(50, seed=0)


def test_variants_drawn_from_voice_style_grid():
    grid = {(v, s, 0.0) for v in VOICES for s in STYLES}  # pitch off -> only the 0.0 cell
    assert all(vs in grid for vs in _variants(80, seed=0))


def test_backend_variants_and_synthesize_decode():
    captured = []

    def fake_synth(client, text, voice, *, model):
        captured.append((text, voice, model))
        return _sine_pcm()

    backend = GeminiTtsBackend(client=object(), synthesize=fake_synth)
    variants = backend.variants(10, seed=0)
    assert len(variants) == 10 and all(isinstance(v, Variant) for v in variants)
    assert all(v.voice in VOICES for v in variants)

    wav = backend.synthesize("hey computer", variants[0])
    assert wav.dtype == np.float32 and abs(len(wav) - SAMPLE_RATE) <= 2
    text, voice, model = captured[0]
    assert text.endswith("hey computer") and voice == variants[0].voice  # style prefix optional


def test_backend_writes_and_caches_through_generate_clips(tmp_path):
    calls = []

    def fake_synth(client, text, voice, *, model):
        calls.append(voice)
        return _sine_pcm()

    backend = GeminiTtsBackend(client=object(), synthesize=fake_synth)
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert len(paths) == 8 and len(calls) == 8

    again = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert again == paths and len(calls) == 8  # all cached -> no new synthesis


def test_pitch_axis_off_by_default():
    backend = GeminiTtsBackend(client=object(), synthesize=lambda *a, **k: _sine_pcm())
    vs = backend.variants(80, seed=0)
    assert all("|p" not in v.tag and "pitch" not in v.params for v in vs)  # tags/params unchanged -> cache-compatible


def test_pitch_axis_on_adds_shifted_variants_and_preserves_length():
    def fake_synth(client, text, voice, *, model):
        return _sine_pcm()

    backend = GeminiTtsBackend(client=object(), synthesize=fake_synth, pitch_semitones=(5.0,))
    vs = backend.variants(420, seed=0)  # full grid: 30 voices * 7 styles * (1 + 1 shift) = 420
    pitched = [v for v in vs if "pitch" in v.params]
    assert pitched and all(v.params["pitch"] == 5.0 and v.tag.endswith("|p+5") for v in pitched)
    assert any("pitch" not in v.params for v in vs)

    out = backend.synthesize("hey computer", pitched[0])
    assert out.dtype == np.float32 and abs(len(out) - SAMPLE_RATE) <= 2  # duration preserved through the shift


def test_backend_reraises_synthesis_error(tmp_path):
    def fake_synth(client, text, voice, *, model):
        raise RuntimeError("quota exceeded")

    backend = GeminiTtsBackend(client=object(), synthesize=fake_synth)
    with pytest.raises(RuntimeError, match="quota"):
        generate_clips("hey computer", tmp_path, backend, n_clips=5, seed=0)
