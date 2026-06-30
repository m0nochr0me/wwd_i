import io

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # hf_audio imports mswc -> clips/episodic -> torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.hf_audio import _write_pool, _write_pool_parquet  # noqa: E402


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


def _blob(item) -> bytes:
    if isinstance(item, (bytes, bytearray)):
        return bytes(item)  # raw bytes -> inject a corrupt clip verbatim
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(item * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _write_parquet(path, items, *, col="audio", as_struct=True, row_group_size=2):
    """Write a parquet shard with an Audio-struct (or raw-binary) audio column for the local-read tests."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    blobs = [_blob(it) for it in items]
    if as_struct:
        arr = pa.array(
            [{"bytes": b, "path": None} for b in blobs],
            type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
        )
    else:
        arr = pa.array(blobs, type=pa.large_binary())
    pq.write_table(pa.table({col: arr}), path, row_group_size=row_group_size)


def test_write_pool_parquet_keeps_all_and_names_sequentially(tmp_path):
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [1.0] * 4, row_group_size=2)  # 2 row groups -> exercises per-row-group iteration
    kept, skipped = _write_pool_parquet(
        [path], tmp_path / "out", n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=0, seed=0
    )
    assert (kept, skipped) == (4, 0)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.wav")) == [f"{i:05d}.wav" for i in range(4)]


def test_write_pool_parquet_skips_short_and_corrupt(tmp_path):
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [0.2, b"notaudio", 1.0], row_group_size=2)
    kept, skipped = _write_pool_parquet(
        [path], tmp_path / "out", n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=0, seed=0
    )
    assert (kept, skipped) == (1, 2)  # short + corrupt skipped


def test_write_pool_parquet_stops_at_n_clips(tmp_path):
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [1.0] * 8, row_group_size=3)
    kept, _ = _write_pool_parquet(
        [path], tmp_path / "out", n_clips=3, audio_key="audio", min_seconds=1.0, max_stream=0, seed=0
    )
    assert kept == 3


def test_write_pool_parquet_max_stream_caps_consultations(tmp_path):
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [1.0] * 10, row_group_size=10)
    kept, _ = _write_pool_parquet(
        [path], tmp_path / "out", n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=2, seed=0
    )
    assert kept == 2  # only 2 rows consulted


def test_write_pool_parquet_binary_audio_column(tmp_path):
    # audio stored as a plain binary column named `flac` (not an Audio struct); auto-detected by hint.
    path = tmp_path / "shard.parquet"
    _write_parquet(path, [1.0, 1.0], col="flac", as_struct=False, row_group_size=1)
    kept, skipped = _write_pool_parquet(
        [path], tmp_path / "out", n_clips=10, audio_key="audio", min_seconds=1.0, max_stream=0, seed=0
    )
    assert (kept, skipped) == (2, 0)
