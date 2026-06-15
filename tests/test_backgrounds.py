import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.backgrounds import find_audio, load_background_pool


def test_find_audio_recursive(tmp_path):
    (tmp_path / "a").mkdir()
    sf.write(tmp_path / "a" / "x.wav", np.zeros(SAMPLE_RATE, np.float32), SAMPLE_RATE)
    sf.write(tmp_path / "y.flac", np.zeros(SAMPLE_RATE, np.float32), SAMPLE_RATE)
    (tmp_path / "note.txt").write_text("nope")
    assert {p.name for p in find_audio(tmp_path)} == {"x.wav", "y.flac"}


def test_load_pool_caps_skips_and_filters(tmp_path):
    sf.write(tmp_path / "stereo.wav", np.zeros((88200, 2), np.float32), 44100)  # 2 s, 44.1k stereo
    sf.write(tmp_path / "short.wav", np.zeros(SAMPLE_RATE // 2, np.float32), SAMPLE_RATE)  # 0.5 s -> filtered
    (tmp_path / "broken.wav").write_bytes(b"not audio")  # -> skipped
    sf.write(tmp_path / "ok.wav", np.zeros(SAMPLE_RATE * 2, np.float32), SAMPLE_RATE)

    pool = load_background_pool(tmp_path, max_clips=10, min_seconds=1.0)
    assert len(pool) == 2  # stereo (downmixed+resampled) + ok
    assert all(w.ndim == 1 and w.dtype == np.float32 for w in pool)
    assert len(load_background_pool(tmp_path, max_clips=1)) == 1  # cap respected
