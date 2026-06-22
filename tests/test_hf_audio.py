import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # hf_audio imports mswc -> clips/episodic -> torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.hf_audio import _write_pool  # noqa: E402


def _ex(seconds: float) -> dict:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE, format="WAV")
    return {"audio": {"bytes": buf.getvalue(), "path": None}}


def test_write_pool_stops_at_n_and_names_sequentially(tmp_path):
    examples = [_ex(1.0) for _ in range(5)] + [_ex(0.2)] + [_ex(1.0) for _ in range(5)]
    kept, _ = _write_pool(examples, tmp_path, n_clips=3, audio_key="audio", min_seconds=1.0, max_stream=0)
    assert kept == 3  # stops once n_clips reached
    assert sorted(p.name for p in tmp_path.glob("*.wav")) == ["00000.wav", "00001.wav", "00002.wav"]


def test_write_pool_skips_short_and_corrupt(tmp_path):
    examples = [_ex(0.2), {"audio": {"bytes": b"notaudio"}}, _ex(1.0)]
    kept, skipped = _write_pool(examples, tmp_path, n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=0)
    assert (kept, skipped) == (1, 2)  # short + corrupt skipped, one valid kept


def test_write_pool_max_stream_caps_consultations(tmp_path):
    examples = [_ex(1.0) for _ in range(10)]
    kept, _ = _write_pool(examples, tmp_path, n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=2)
    assert kept == 2  # only the first 2 examples consulted


def test_write_pool_decodes_raw_bytes_under_custom_key(tmp_path):
    # WebDataset (e.g. 0x3/vocal-bursts) yields FLAC bytes directly, not a {bytes|path} dict.
    buf = io.BytesIO()
    sf.write(buf, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, format="FLAC")
    examples = [{"flac": buf.getvalue()}]
    kept, skipped = _write_pool(examples, tmp_path, n_clips=10, audio_key="flac", min_seconds=0.4, max_stream=0)
    assert (kept, skipped) == (1, 0)


def test_write_pool_wrong_audio_key_raises(tmp_path):
    with pytest.raises(KeyError):  # surfaces a bad --audio-key instead of skipping every clip
        _write_pool([_ex(1.0)], tmp_path, n_clips=1, audio_key="flac", min_seconds=1.0, max_stream=0)
