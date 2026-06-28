import io

import numpy as np
import pytest
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.groq_tts import (
    STYLES,
    VOICES,
    GroqTtsBackend,
    _prompt,
    _variants,
    _wav_to_float16k,
)
from wwd_i.data.tts import Variant, generate_clips


def _sine_wav(seconds: float = 1.0, rate: int = 24_000) -> bytes:
    """A WAV byte blob (header + s16le mono), as Orpheus returns."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_wav_decode_resamples_to_16k():
    wav = _wav_to_float16k(_sine_wav(seconds=1.0, rate=24_000))
    assert wav.dtype == np.float32
    assert abs(len(wav) - SAMPLE_RATE) <= 2  # 24 kHz -> 16 kHz, one second
    assert np.max(np.abs(wav)) <= 1.0


def test_wav_decode_reads_rate_from_header():
    # A different header rate must still land at SAMPLE_RATE (rate read from the WAV, not assumed).
    wav = _wav_to_float16k(_sine_wav(seconds=1.0, rate=48_000))
    assert abs(len(wav) - SAMPLE_RATE) <= 2


def test_prompt_neutral_vs_styled():
    assert _prompt("hey computer", "") == "hey computer"
    assert _prompt("hey computer", "[cheerful]") == "[cheerful] hey computer"


def test_variants_deterministic_and_count():
    assert _variants(50, seed=0) == _variants(50, seed=0)
    assert len(_variants(50, seed=0)) == 50
    assert _variants(50, seed=1) != _variants(50, seed=0)


def test_variants_drawn_from_voice_style_grid():
    grid = {(v, s) for v in VOICES for s in STYLES}
    assert all(vs in grid for vs in _variants(80, seed=0))


def test_backend_variants_and_synthesize_decode():
    captured = []

    def fake_synth(client, text, voice, *, model):
        captured.append((text, voice, model))
        return _sine_wav()

    backend = GroqTtsBackend(client=object(), synthesize=fake_synth)
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
        return _sine_wav()

    backend = GroqTtsBackend(client=object(), synthesize=fake_synth)
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert len(paths) == 8 and len(calls) == 8

    again = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert again == paths and len(calls) == 8  # all cached -> no new synthesis


def test_backend_reraises_synthesis_error(tmp_path):
    def fake_synth(client, text, voice, *, model):
        raise RuntimeError("quota exceeded")

    backend = GroqTtsBackend(client=object(), synthesize=fake_synth)
    with pytest.raises(RuntimeError, match="quota"):
        generate_clips("hey computer", tmp_path, backend, n_clips=5, seed=0)
