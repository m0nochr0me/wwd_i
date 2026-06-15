import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # mswc imports clips/episodic, which import torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.mswc import _Subset, _to_16k_mono, mswc_samplers  # noqa: E402


def test_to_16k_mono_resamples_and_downmixes():
    stereo_48k = np.zeros((48000, 2), dtype=np.float32)  # 1 s, 48 kHz, stereo
    out = _to_16k_mono({"array": stereo_48k, "sampling_rate": 48000})
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) - SAMPLE_RATE) <= 2  # downmixed + resampled to ~16k samples

    mono_16k = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert len(_to_16k_mono({"array": mono_16k, "sampling_rate": SAMPLE_RATE})) == SAMPLE_RATE


def test_subset_caps_clips_per_word():
    sub = _Subset(n_words=3, clips_per_word=2)
    assert [sub.take("a") for _ in range(4)] == [0, 1, None, None]  # >2 clips dropped
    assert not sub.done


def test_subset_caps_vocabulary_and_reports_done():
    sub = _Subset(n_words=2, clips_per_word=1)
    assert sub.take("a") == 0
    assert sub.take("b") == 0
    assert sub.take("c") is None  # third word ignored once vocabulary is full
    assert sub.done  # both words reached their clip quota


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
