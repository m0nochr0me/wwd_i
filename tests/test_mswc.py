import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # mswc imports clips/episodic, which import torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.mswc import _decode_16k_mono, _WordSelector, mswc_samplers  # noqa: E402


def test_decode_16k_mono_from_bytes():
    import io

    buf = io.BytesIO()
    sf.write(buf, np.zeros((48000, 2), dtype=np.float32), 48000, format="WAV")  # 1 s, 48 kHz, stereo
    out = _decode_16k_mono({"bytes": buf.getvalue(), "path": None})
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) - SAMPLE_RATE) <= 2  # decoded + downmixed + resampled to ~16k

    buf16 = io.BytesIO()
    sf.write(buf16, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, format="WAV")
    assert len(_decode_16k_mono({"bytes": buf16.getvalue()})) == SAMPLE_RATE


def test_word_selector_deterministic_and_alphabet_spread():
    sel = _WordSelector(n_words=100, vocab_total=1000, seed=0)
    assert sel.keep("hello") == sel.keep("hello")  # deterministic

    words = [f"{c}{i}" for c in "abcdefghijklmnopqrstuvwxyz" for i in range(40)]  # ~1040 words, a..z
    kept = [w for w in words if sel.keep(w)]
    assert 50 < len(kept) < 160  # ~100 of ~1040 (loose statistical bounds)
    assert len({w[0] for w in kept}) > 10  # spread across many starting letters, not just 'a'


def test_word_selector_keeps_all_when_target_exceeds_total():
    sel = _WordSelector(n_words=1000, vocab_total=100, seed=0)
    assert all(sel.keep(f"w{i}") for i in range(50))  # threshold >= modulus -> keep everything


def test_mswc_samplers_random_held_out(tmp_path):
    for w in range(6):
        for k in range(8):
            p = tmp_path / f"w{w}" / f"{k}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    train, held = mswc_samplers(tmp_path, n_held_out=2, seed=0)
    assert len(held.labels) == 2
    assert len(train.labels) == 4
    assert set(train.labels).isdisjoint(held.labels)


def test_mswc_samplers_augment_train_only(tmp_path):
    for w in range(6):
        for k in range(8):
            p = tmp_path / f"w{w}" / f"{k}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    train, held = mswc_samplers(tmp_path, n_held_out=2, seed=0, augment=lambda w: w + 1.0)
    assert np.allclose(train.episode(4, 1, 1).numpy(), 1.0)  # augment fired on training clips
    assert np.allclose(held.episode(2, 1, 1).numpy(), 0.0)  # probe clips untouched
