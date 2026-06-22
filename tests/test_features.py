from typing import cast

import numpy as np
import pytest

pytest.importorskip("torchaudio")  # mel front-end needs the `train` group (torch/torchaudio)

from wwd_i.config import HOP_LENGTH, N_MELS, SAMPLE_RATE, WIN_LENGTH  # noqa: E402
from wwd_i.features.melspec import compute_logmel, export_onnx  # noqa: E402
from wwd_i.features.streaming import MelStreamer  # noqa: E402


def _signal(seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.linspace(0, seconds, int(seconds * SAMPLE_RATE), endpoint=False, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 330 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
    noise = 0.01 * rng.standard_normal(t.shape).astype(np.float32)
    return (tone + noise).astype(np.float32)


def test_logmel_shape():
    sig = _signal(0.5)
    mel = compute_logmel(sig)
    expected_frames = 1 + (len(sig) - WIN_LENGTH) // HOP_LENGTH
    assert mel.shape == (expected_frames, N_MELS)
    assert mel.dtype == np.float32


def test_onnx_parity(tmp_path):
    import onnxruntime as ort

    path = export_onnx(tmp_path / "melspec.onnx")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    sig = _signal(1.0)
    ref = compute_logmel(sig)
    onnx_out = cast(np.ndarray, sess.run(None, {"audio": sig[None, :]})[0])[0]

    assert onnx_out.shape == ref.shape
    assert np.max(np.abs(onnx_out - ref)) < 1e-3


def test_streaming_matches_batch():
    sig = _signal(1.3)
    ref = compute_logmel(sig)

    streamer = MelStreamer(compute_logmel)
    outs = []
    for pos in range(0, len(sig), 1280):  # 80 ms stream frames
        outs.append(streamer.push(sig[pos : pos + 1280]))
    streamed = np.concatenate([c for c in outs if len(c)], axis=0)

    assert streamed.shape == ref.shape
    assert np.max(np.abs(streamed - ref)) < 1e-4
