import numpy as np

from wwd_i.config import FRAME_SAMPLES
from wwd_i.export import build_identity_model
from wwd_i.runtime import load_session, run_stream


def test_identity_stream_roundtrip(tmp_path):
    session = load_session(build_identity_model(tmp_path / "identity.onnx"))

    frames = [np.full(FRAME_SAMPLES, i, dtype=np.float32) for i in range(5)]
    seen: list[np.ndarray] = []
    stats = run_stream(session, iter(frames), on_output=lambda _i, out: seen.append(out))

    assert stats.frames == 5
    assert stats.audio_ms == 5 * 80
    assert len(seen) == 5
    for src, out in zip(frames, seen, strict=True):
        np.testing.assert_array_equal(src, out)
    assert stats.real_time_factor >= 0.0


def test_run_stream_empty(tmp_path):
    session = load_session(build_identity_model(tmp_path / "identity.onnx"))
    stats = run_stream(session, iter([]))
    assert stats.frames == 0
    assert stats.audio_ms == 0
    assert stats.real_time_factor == 0.0
