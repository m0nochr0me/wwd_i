import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # mswc imports clips/episodic, which import torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.mswc import _decode_16k_mono, _Subset, mswc_samplers  # noqa: E402


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


def test_subset_caps_clips_per_word():
    sub = _Subset(n_words=3, clips_per_word=2)
    assert [sub.take("a") for _ in range(4)] == [0, 1, None, None]  # >2 clips dropped
    assert not sub.done


def test_subset_done_when_n_words_full_no_vocab_cap():
    sub = _Subset(n_words=2, clips_per_word=1)
    assert sub.take("a") == 0
    assert not sub.done
    assert sub.take("b") == 0  # second full word -> done
    assert sub.done
    assert sub.take("c") == 0  # still accepts new words (no vocabulary cap)


def test_subset_selected_keeps_fullest():
    sub = _Subset(n_words=2, clips_per_word=3)
    for _ in range(3):
        sub.take("a")
    for _ in range(3):
        sub.take("b")
    sub.take("c")  # under-filled, must be dropped
    assert sub.selected() == ["a", "b"]


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
