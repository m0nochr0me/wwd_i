"""Phase-5 streaming engine tests (torch-free).

The mel + backbone stages are the packaged ONNX models; the head is a tiny
synthetic stateful ONNX built here with ``onnx.helper`` so the suite needs no
trained artifact and no torch. The head's next state depends on BOTH the
embedding and the previous state, so a dropped-state or mis-fed-embedding bug in
the engine changes its probability sequence and the parity test catches it.
"""

import numpy as np
import onnx
import pytest
import soundfile as sf
from onnx import TensorProto, helper, numpy_helper

from wwd_i.audio.io import load_wav
from wwd_i.config import FRAME_SAMPLES, HOP_LENGTH, MEL_FRAMES_PER_FRAME, SAMPLE_RATE
from wwd_i.runtime import WakeWordEngine, detect_file
from wwd_i.runtime.engine import WINDOW

D, H = 96, 48
HOP_S = MEL_FRAMES_PER_FRAME * HOP_LENGTH / SAMPLE_RATE  # 0.08 s — one embedding per 80 ms frame


def _toy_head(path):
    """A minimal stateful GRU-ish head with the export contract
    (embedding[1,1,D], h0[1,1,H]) -> (prob[1,1], hn[1,1,H]). State persists
    (Wh=0.9·I) and a positive bias drives P(wake) high, so it both carries
    history and reliably crosses a 0.5 threshold."""
    rng = np.random.default_rng(0)
    arrays = {
        "Wx": (rng.standard_normal((D, H)) * 0.05).astype(np.float32),
        "Wh": (0.9 * np.eye(H)).astype(np.float32),
        "bh": np.full(H, 0.3, np.float32),
        "Wo": np.full((H, 1), 0.5, np.float32),
        "bo": np.array([0.5], np.float32),
        "se": np.array([-1, D], np.int64),
        "sh": np.array([-1, H], np.int64),
        "shn": np.array([1, 1, H], np.int64),
    }
    inits = [numpy_helper.from_array(v, k) for k, v in arrays.items()]
    nodes = [
        helper.make_node("Reshape", ["embedding", "se"], ["e2"]),
        helper.make_node("Reshape", ["h0", "sh"], ["h2"]),
        helper.make_node("MatMul", ["e2", "Wx"], ["a"]),
        helper.make_node("MatMul", ["h2", "Wh"], ["b"]),
        helper.make_node("Add", ["a", "b"], ["ab"]),
        helper.make_node("Add", ["ab", "bh"], ["pre"]),
        helper.make_node("Tanh", ["pre"], ["hn2"]),
        helper.make_node("MatMul", ["hn2", "Wo"], ["wl"]),
        helper.make_node("Add", ["wl", "bo"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["prob"]),
        helper.make_node("Reshape", ["hn2", "shn"], ["hn"]),
    ]
    graph = helper.make_graph(
        nodes,
        "toy_head",
        [
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 1, D]),
            helper.make_tensor_value_info("h0", TensorProto.FLOAT, [1, 1, H]),
        ],
        [
            helper.make_tensor_value_info("prob", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("hn", TensorProto.FLOAT, [1, 1, H]),
        ],
        inits,
    )
    model = helper.make_model(graph, producer_name="wwd_i_test", opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def _engine(tmp_path, *, threshold, refractory):
    return WakeWordEngine(_toy_head(tmp_path / "toy_head.onnx"), threshold=threshold, refractory_s=refractory)


def _noise(n_frames, seed):
    return np.random.default_rng(seed).standard_normal(FRAME_SAMPLES * n_frames).astype(np.float32) * 0.1


def test_streaming_matches_batch(tmp_path):
    """Phase-5 gate: detections are identical whether audio arrives in one block
    or as a stream of 80 ms frames. Threshold below all probs + zero refractory
    turns every hop into a detection, so this compares the full per-hop sequence."""
    sig = _noise(60, seed=3)

    batch = _engine(tmp_path, threshold=-1.0, refractory=0.0).push(sig)

    eng = _engine(tmp_path, threshold=-1.0, refractory=0.0)
    stream = [d for i in range(60) for d in eng.push(sig[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])]

    assert len(batch) == len(stream) > 0
    np.testing.assert_array_equal([d.time_s for d in batch], [d.time_s for d in stream])  # deterministic time grid
    diff = np.max(np.abs([d.score for d in batch] - np.array([d.score for d in stream])))
    assert diff < 1e-4, f"seam/state mismatch: {diff}"  # GRU state carried identically across chunk boundaries


def test_refractory_spacing(tmp_path):
    """Above-threshold hops are debounced to at most one detection per refractory window."""
    eng = _engine(tmp_path, threshold=0.5, refractory=1.0)
    dets = eng.push(_noise(80, seed=5))

    assert len(dets) >= 2
    assert all(d.score >= 0.5 for d in dets)
    gaps = np.diff([d.time_s for d in dets])
    assert np.all(gaps >= 1.0 - 1e-9)


def test_detection_time_grid(tmp_path):
    """Detection timestamps land on the (k·8 + WINDOW)·hop grid — window-end times."""
    dets = _engine(tmp_path, threshold=-1.0, refractory=0.0).push(_noise(30, seed=9))

    base = WINDOW * HOP_LENGTH / SAMPLE_RATE
    expected = [base + k * HOP_S for k in range(len(dets))]
    np.testing.assert_allclose([d.time_s for d in dets], expected, atol=1e-9)
    assert abs(HOP_S - 0.08) < 1e-12  # one embedding per 80 ms


def test_detect_file_resets_and_matches_whole_push(tmp_path):
    """detect_file resets state then scores the whole file == a fresh whole-file push."""
    sig = _noise(50, seed=11)
    wav = tmp_path / "clip.wav"
    sf.write(str(wav), sig, SAMPLE_RATE, subtype="FLOAT")

    eng = _engine(tmp_path, threshold=0.5, refractory=1.0)
    eng.push(_noise(10, seed=99))  # dirty the state first; detect_file must reset
    file_dets = detect_file(eng, wav)

    ref = _engine(tmp_path, threshold=0.5, refractory=1.0).push(load_wav(wav))
    assert [d.time_s for d in file_dets] == [d.time_s for d in ref]


def test_window_matches_trainer():
    """The runtime window must equal the window the head was trained on."""
    pytest.importorskip("torch")
    from wwd_i.train.train_head import WINDOW as TRAIN_WINDOW

    assert WINDOW == TRAIN_WINDOW


def test_packaged_melspec_onnx_parity():
    """The committed melspec.onnx matches the torch front-end and is self-contained."""
    pytest.importorskip("torch")
    from importlib.resources import files

    import onnxruntime as ort

    from wwd_i.features.melspec import compute_logmel

    path = files("wwd_i.models") / "melspec.onnx"
    assert not path.with_suffix(".onnx.data").exists()  # inline weights, ships as one file
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    rng = np.random.default_rng(2)
    for n in (SAMPLE_RATE, int(1.5 * SAMPLE_RATE)):  # dynamic sample length
        sig = rng.standard_normal(n).astype(np.float32)
        err = np.max(np.abs(sess.run(None, {name: sig[None]})[0][0] - compute_logmel(sig)))
        assert err < 1e-3
