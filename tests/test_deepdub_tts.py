import io

import numpy as np
import pytest
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.deepdub_tts import (
    TEMPERATURES,
    VARIANCES,
    DeepdubTtsBackend,
    _decode_to_float16k,
    _variants,
)
from wwd_i.data.tts import Variant, generate_clips

VOICES = ("voice-a", "voice-b")


def _wav_bytes(seconds: float = 1.0, rate: int = 24_000) -> bytes:
    """Soundfile-readable audio bytes; the decode path is container-agnostic (real
    backend gets mp3, validated live)."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_resamples_to_16k():
    wav = _decode_to_float16k(_wav_bytes(seconds=1.0, rate=24_000))
    assert wav.dtype == np.float32
    assert abs(len(wav) - SAMPLE_RATE) <= 2
    assert np.max(np.abs(wav)) <= 1.0


def test_decode_reads_rate_from_header():
    wav = _decode_to_float16k(_wav_bytes(seconds=1.0, rate=48_000))
    assert abs(len(wav) - SAMPLE_RATE) <= 2


def test_variants_deterministic_and_count():
    assert _variants(50, VOICES, seed=0) == _variants(50, VOICES, seed=0)
    assert len(_variants(50, VOICES, seed=0)) == 50
    assert _variants(50, VOICES, seed=1) != _variants(50, VOICES, seed=0)


def test_variants_distinct_seed_per_clip():
    # Few voices, many clips: cells repeat but the per-clip seed keeps every variant distinct.
    out = _variants(40, ("solo",), seed=5)
    seeds = [s for _, _, _, s in out]
    assert seeds == list(range(5, 45))  # seed + i
    tags = {(v, t, var, s) for v, t, var, s in out}
    assert len(tags) == 40  # all distinct despite one voice


def test_variants_drawn_from_grid():
    for v, t, var, _ in _variants(60, VOICES, seed=0):
        assert v in VOICES and t in TEMPERATURES and var in VARIANCES


def test_backend_variants_and_synthesize_decode():
    captured = []

    def fake_synth(client, text, vid, *, model, temperature, variance, seed):
        captured.append((text, vid, model, temperature, variance, seed))
        return _wav_bytes()

    backend = DeepdubTtsBackend(client=object(), voice_prompt_ids=VOICES, synthesize=fake_synth)
    variants = backend.variants(10, seed=0)
    assert len(variants) == 10 and all(isinstance(v, Variant) for v in variants)
    assert all(v.voice in VOICES for v in variants)

    wav = backend.synthesize("hey computer", variants[0])
    assert wav.dtype == np.float32 and abs(len(wav) - SAMPLE_RATE) <= 2
    text, vid, model, temp, var, seed = captured[0]
    assert text == "hey computer" and vid == variants[0].voice
    assert temp in TEMPERATURES and var in VARIANCES  # prosody knobs threaded through


def test_backend_writes_and_caches_through_generate_clips(tmp_path):
    calls = []

    def fake_synth(client, text, vid, *, model, temperature, variance, seed):
        calls.append((vid, seed))
        return _wav_bytes()

    backend = DeepdubTtsBackend(client=object(), voice_prompt_ids=VOICES, synthesize=fake_synth)
    paths = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert len(paths) == 8 and len(calls) == 8  # 8 distinct (seed makes each unique)

    again = generate_clips("hey computer", tmp_path, backend, n_clips=8, seed=0)
    assert again == paths and len(calls) == 8  # all cached -> no new synthesis


def test_backend_reraises_synthesis_error(tmp_path):
    def fake_synth(client, text, vid, *, model, temperature, variance, seed):
        raise RuntimeError("quota exceeded")

    backend = DeepdubTtsBackend(client=object(), voice_prompt_ids=VOICES, synthesize=fake_synth)
    with pytest.raises(RuntimeError, match="quota"):
        generate_clips("hey computer", tmp_path, backend, n_clips=5, seed=0)


def test_empty_voice_ids_raises():
    backend = DeepdubTtsBackend(client=object(), voice_prompt_ids=())
    with pytest.raises(RuntimeError, match="voice_prompt_ids"):
        backend.variants(5, seed=0)
