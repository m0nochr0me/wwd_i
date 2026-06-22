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


def _flac(seconds: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE, format="FLAC")
    return buf.getvalue()


def test_write_pool_decodes_raw_bytes_under_custom_key(tmp_path):
    # WebDataset (e.g. 0x3/vocal-bursts) yields FLAC bytes directly, not a {bytes|path} dict.
    examples = [{"flac": _flac(1.0)}]
    kept, skipped = _write_pool(examples, tmp_path, n_clips=10, audio_key="flac", min_seconds=0.4, max_stream=0)
    assert (kept, skipped) == (1, 0)


def test_write_pool_auto_detects_audio_column(tmp_path):
    # audio under `flac` with json alongside; default --audio-key audio must pick flac, not json.
    examples = [{"flac": _flac(1.0), "__key__": "a", "json": {"label": "cough"}}]
    kept, skipped = _write_pool(examples, tmp_path, n_clips=10, audio_key="audio", min_seconds=0.4, max_stream=0)
    assert (kept, skipped) == (1, 0)  # resolves to `flac` instead of crashing on KeyError: 'audio'


def test_write_pool_skips_metadata_only_examples(tmp_path):
    # 0x3/vocal-bursts streams audio and JSON as SEPARATE, unpaired examples; metadata-only ones
    # carry no audio and must be skipped (not crash, not selected as the audio column).
    examples = [
        {"json": {"label": "breath"}, "__key__": "a", "__url__": "Breath.tar.gz"},  # metadata-only
        {"flac": _flac(1.0), "__key__": "b", "__url__": "Breath.tar.gz"},  # audio
    ]
    kept, skipped = _write_pool(examples, tmp_path, n_clips=10, audio_key="audio", min_seconds=0.4, max_stream=0)
    assert (kept, skipped) == (1, 1)
