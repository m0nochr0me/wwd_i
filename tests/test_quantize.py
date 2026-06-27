"""Phase-6 int8 quantization tests (torch-free).

Exercises the real packaged ``backbone.onnx`` — the dynamo-exported model that crashes
ORT's quantizer without the ``value_info`` strip — plus the head path (via the synthetic
``_toy_head`` from the engine tests) and the label-free regression guard. No torch, no
trained head artifact.
"""

from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import soundfile as sf
from onnxruntime.quantization import QuantType, quantize_dynamic
from test_engine import _toy_head  # tests/ is not a package; pytest prepend mode makes this top-level

from wwd_i.config import SAMPLE_RATE
from wwd_i.export import quantize_dynamic_model, quantize_static_model
from wwd_i.export.compare import compare
from wwd_i.export.quantize import MelCalibrationReader


def _backbone() -> str:
    return str(files("wwd_i.models") / "backbone.onnx")


def _write_calib_wavs(d, n=8, seed=0):
    """A handful of ~1.5 s -20 dBFS noise clips to calibrate against (torch-free)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        sig = (rng.standard_normal(int(1.5 * SAMPLE_RATE)) * 0.1).astype(np.float32)
        sf.write(str(d / f"calib_{i}.wav"), sig, SAMPLE_RATE, subtype="FLOAT")
    return d


def _onmanifold_windows(calib_dir) -> np.ndarray:
    """Backbone-input windows on the inference manifold (same path the reader uses)."""
    bb_in = ort.InferenceSession(_backbone(), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    reader = MelCalibrationReader(calib_dir, bb_in, max_windows=64)
    out = []
    while (item := reader.get_next()) is not None:
        out.append(item[bb_in])
    return np.concatenate(out, axis=0)  # [N, WINDOW, n_mels]


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def test_strip_fixes_dynamo_shape_inference_crash(tmp_path):
    """Raw quantize_dynamic on the dynamo-exported backbone aborts in onnx shape
    inference (stale value_info); our wrapper strips it first and succeeds. Pins the
    gotcha and its fix together."""
    with pytest.raises(Exception):  # noqa: B017 - onnx C++ InferenceError, not a clean Python type
        quantize_dynamic(_backbone(), str(tmp_path / "raw.onnx"), weight_type=QuantType.QInt8)

    out = quantize_dynamic_model(_backbone(), tmp_path / "bb.int8.onnx")
    assert out.exists()


def test_dynamic_backbone_smaller_singlefile_and_close(tmp_path):
    """Dynamic int8 backbone: self-contained (no sidecar), smaller, quantizes the convs,
    and tracks the float embedding (cosine ~ unit)."""
    out = quantize_dynamic_model(_backbone(), tmp_path / "bb.int8.onnx")
    assert not out.with_suffix(".onnx.data").exists()  # single-file invariant
    assert out.stat().st_size < Path(_backbone()).stat().st_size

    ops = Counter(n.op_type for n in onnx.load(str(out)).graph.node)
    assert ops["ConvInteger"] >= 1  # the BC-ResNet convs are quantized, not just the proj

    calib = _onmanifold_windows(_write_calib_wavs(tmp_path))
    f = ort.InferenceSession(_backbone(), providers=["CPUExecutionProvider"])
    q = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ef = cast(np.ndarray, f.run(None, {f.get_inputs()[0].name: calib})[0])
    eq = cast(np.ndarray, q.run(None, {q.get_inputs()[0].name: calib})[0])
    assert _cosine(ef, eq).min() > 0.98


def test_static_qdq_backbone_is_qdq_singlefile_and_close(tmp_path):
    """Static QDQ backbone: QDQ format (Q/DQ around the float convs — XNNPACK-ready),
    self-contained, and embedding stays close on on-manifold calibration."""
    calib_dir = _write_calib_wavs(tmp_path)
    out = quantize_static_model(_backbone(), tmp_path / "bb.qdq.onnx", calib_dir)
    assert not out.with_suffix(".onnx.data").exists()

    ops = Counter(n.op_type for n in onnx.load(str(out)).graph.node)
    assert ops["QuantizeLinear"] >= 1 and ops["DequantizeLinear"] >= 1 and ops["Conv"] >= 1  # QDQ, conv left float

    calib = _onmanifold_windows(calib_dir)
    f = ort.InferenceSession(_backbone(), providers=["CPUExecutionProvider"])
    q = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ef = cast(np.ndarray, f.run(None, {f.get_inputs()[0].name: calib})[0])
    eq = cast(np.ndarray, q.run(None, {q.get_inputs()[0].name: calib})[0])
    assert _cosine(ef, eq).min() > 0.95  # static is a touch looser than dynamic; still well above noise


def test_static_quant_preserves_loudness_invariance(tmp_path):
    """The z-score and L2-normalize ops stay float (only Conv/Gemm are quantized), so the
    backbone is still invariant to a uniform loudness change (+20 dB == +ln(100)/mel bin)."""
    calib_dir = _write_calib_wavs(tmp_path)
    out = quantize_static_model(_backbone(), tmp_path / "bb.qdq.onnx", calib_dir)
    calib = _onmanifold_windows(calib_dir)
    q = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    name = q.get_inputs()[0].name
    base = cast(np.ndarray, q.run(None, {name: calib})[0])
    loud = cast(np.ndarray, q.run(None, {name: (calib + 4.605).astype(np.float32)})[0])
    assert _cosine(base, loud).min() > 0.99


def test_dynamic_head_quantizes_and_runs(tmp_path):
    """Dynamic quant of a (clean, hand-built) head: smaller, self-contained, still emits a
    valid probability for the (embedding, state) -> (prob, state) contract."""
    head = _toy_head(tmp_path / "toy_head.onnx")
    out = quantize_dynamic_model(head, tmp_path / "toy.int8.onnx")
    assert out.exists() and not out.with_suffix(".onnx.data").exists()
    assert out.stat().st_size < head.stat().st_size

    s = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in s.get_inputs()}
    d = ins["embedding"].shape[-1]
    h = ins["h0"].shape[-1]
    res = s.run(None, {"embedding": np.zeros((1, 1, d), np.float32), "h0": np.zeros((1, 1, h), np.float32)})
    prob, hn = cast(np.ndarray, res[0]), cast(np.ndarray, res[1])
    assert prob.shape == (1, 1) and 0.0 <= float(prob.ravel()[0]) <= 1.0
    assert hn.shape == (1, 1, h)


def test_compare_guard_reports_divergence(tmp_path):
    """The label-free guard streams the same audio through float and int8 backbones and
    reports the per-hop P(wake) shift; identical models -> zero divergence (sanity)."""
    head = _toy_head(tmp_path / "toy_head.onnx")
    int8_bb = quantize_dynamic_model(_backbone(), tmp_path / "bb.int8.onnx")
    audio = (np.random.default_rng(0).standard_normal(SAMPLE_RATE * 4) * 0.1).astype(np.float32)

    r = compare(head, audio, int8_backbone=int8_bb)
    assert r["hops"] > 0 and 0.0 <= r["max_abs"] <= 1.0 and r["rms"] <= r["max_abs"]

    same = compare(head, audio)  # float vs float -> no divergence
    assert same["max_abs"] == 0.0
