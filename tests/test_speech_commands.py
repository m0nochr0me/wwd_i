import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # speech_commands imports EpisodicSampler, which imports torch

from wwd_i.audio.io import load_wav  # noqa: E402
from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.speech_commands import _fixed_length, _index_clips  # noqa: E402


def _write_clip(path, n=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(n, dtype=np.float32), SAMPLE_RATE)


def test_index_clips_groups_by_word_and_skips_noise(tmp_path):
    root = tmp_path / "SpeechCommands" / "speech_commands_v0.02"
    for label in ("yes", "no"):
        for k in range(3):
            _write_clip(root / label / f"clip{k}.wav")
    _write_clip(root / "_background_noise_" / "pink.wav")

    index = _index_clips(tmp_path)  # globs recursively from the dataset root
    assert set(index) == {"yes", "no"}  # background-noise folder excluded
    assert all(len(paths) == 3 for paths in index.values())
    assert all(p.suffix == ".wav" for paths in index.values() for p in paths)


def test_fixed_length_pads_and_crops():
    assert len(_fixed_length(np.ones(15600, dtype=np.float32), SAMPLE_RATE)) == SAMPLE_RATE
    assert len(_fixed_length(np.ones(20000, dtype=np.float32), SAMPLE_RATE)) == SAMPLE_RATE


def test_clip_loads_via_soundfile_to_fixed_length(tmp_path):
    # The real loader path: soundfile (not torchaudio/torchcodec) -> fixed length.
    clip = tmp_path / "short.wav"
    _write_clip(clip, n=15600)  # a sub-1 s clip, as some Speech Commands files are
    out = _fixed_length(load_wav(clip), SAMPLE_RATE)
    assert out.shape == (SAMPLE_RATE,)
    assert out.dtype == np.float32
