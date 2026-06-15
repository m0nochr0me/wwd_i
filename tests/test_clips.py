import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # clips imports EpisodicSampler, which imports torch

from wwd_i.audio.io import load_wav  # noqa: E402
from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.clips import fixed_length, index_clips, split_samplers  # noqa: E402


def _write_clip(path, n=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(n, dtype=np.float32), SAMPLE_RATE)


def _word_dir(root, words, n_per=6):
    for word in words:
        for k in range(n_per):
            _write_clip(root / word / f"{k}.wav")


def test_index_clips_groups_by_word_and_skips_noise(tmp_path):
    _word_dir(tmp_path, ("yes", "no"), n_per=3)
    _write_clip(tmp_path / "_background_noise_" / "pink.wav")

    index = index_clips(tmp_path)
    assert set(index) == {"yes", "no"}  # background-noise folder excluded
    assert all(len(paths) == 3 for paths in index.values())
    assert all(p.suffix == ".wav" for paths in index.values() for p in paths)


def test_fixed_length_pads_and_crops():
    assert len(fixed_length(np.ones(15600, dtype=np.float32))) == SAMPLE_RATE
    assert len(fixed_length(np.ones(20000, dtype=np.float32))) == SAMPLE_RATE


def test_clip_loads_via_soundfile_to_fixed_length(tmp_path):
    # The real loader path: soundfile (not torchaudio/torchcodec) -> fixed length.
    clip = tmp_path / "short.wav"
    _write_clip(clip, n=15600)
    out = fixed_length(load_wav(clip))
    assert out.shape == (SAMPLE_RATE,)
    assert out.dtype == np.float32


def test_split_samplers_disjoint_and_limited(tmp_path):
    _word_dir(tmp_path, ("alpha", "beta", "gamma", "delta"), n_per=8)
    train, held = split_samplers(index_clips(tmp_path), held_out_words=["gamma", "delta"], limit=5, seed=0)

    assert set(train.labels) == {"alpha", "beta"}
    assert set(held.labels) == {"gamma", "delta"}
    assert all(len(train.items_by_class[w]) == 5 for w in train.labels)  # limit applied
    episode = train.episode(n_way=2, k_shot=2, q_query=2)
    assert episode.shape == (8, SAMPLE_RATE)  # 2 words * (2 + 2) clips
