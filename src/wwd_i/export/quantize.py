"""int8 quantization of the ONNX artifacts (Phase 6).

Native ``onnxruntime.quantization`` (ships in the base onnxruntime wheel, so this is
torch-free and runs in the inference install). The three artifacts are quantized
independently:

* ``backbone.onnx`` and ``<word>_head.onnx``: supported here.
* ``melspec.onnx``: left float on purpose — its DFT + log is numerically touchy and
  already cheap, and quantizing it risks the mel contract every later stage depends on.

Two modes:

* **dynamic** (:func:`quantize_dynamic_model`): weights → int8, activations quantized
  at run time, no calibration data. Biggest *size* win on x86 (backbone 204→127 KB,
  head 102→37 KB) and it does quantize the BC-ResNet convs (``ConvInteger``). Good
  default for a quick win.
* **static QDQ** (:func:`quantize_static_model`): weights *and* activations int8 with a
  small calibration set. Only marginally smaller than float on x86, but QDQ is the
  format the XNNPACK execution provider fuses into int8 kernels on ARM64 (RPi5) — so
  this is the real latency/CPU win on the SBC target. Needs representative −20 dBFS
  audio (:class:`MelCalibrationReader`).

**The dynamo-export gotcha.** ``backbone.onnx`` / the head are exported with torch's
dynamo exporter, which writes stale intermediate ``value_info`` shape annotations. ORT's
quantizer re-runs onnx shape inference and aborts on the conflict
(``Inferred shape and existing shape differ ... (48) vs (96)`` for the backbone,
``(96) vs (144)`` for the GRU head). Stripping ``graph.value_info`` first lets onnx
re-infer cleanly; without it nothing quantizes. :func:`_prepare` does this. (Same family
of dynamo-exporter footgun as the ``external_data=False`` rule in ``models/backbone.py``;
ORT still prints a "consider pre-processing" warning — it is harmless, the necessary prep
is the value_info strip.)

**Never ship an unguarded int8 model** (Phase 6 mandate): re-run the Phase-4 FA/FR gate
against the quantized artifact (``train_head --backbone backbone.int8.onnx``) and/or the
label-free P(wake) divergence check in :mod:`wwd_i.export.compare`.

Run: ``uv run python -m wwd_i.export.quantize <in.onnx> <out.onnx> --mode dynamic``
     ``uv run python -m wwd_i.export.quantize backbone.onnx backbone.int8.onnx --mode static --calib-wavs <dir>``
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from wwd_i.audio.io import load_wav
from wwd_i.config import MEL_FRAMES_PER_FRAME, SAMPLE_RATE
from wwd_i.runtime.engine import AGC_WINDOW_S, WINDOW, _packaged, _RmsNormalizer


def _prepare(src: str | Path, dst: str | Path) -> None:
    """Write ``src`` to ``dst`` with stale dynamo ``value_info`` stripped.

    Required before quantizing our dynamo-exported models: their leftover
    intermediate shape annotations make ORT's quantizer abort in onnx shape
    inference (see module docstring). Stripping them lets onnx re-infer cleanly.
    """
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    onnx.save(model, str(dst))


def _assert_single_file(path: Path) -> None:
    """No external-data sidecar — artifacts must ship as one self-contained file
    (same invariant the float exporters enforce with ``external_data=False``)."""
    sidecar = path.with_suffix(path.suffix + ".data")
    if sidecar.exists():
        raise RuntimeError(
            f"quantization wrote an external-data sidecar {sidecar}; the shipped .onnx must be "
            "self-contained (it is unloadable once moved without the sidecar)"
        )


def quantize_dynamic_model(src: str | Path, out: str | Path, *, per_channel: bool = False) -> Path:
    """Dynamic int8 quantization (weights → int8, no calibration). Returns ``out``.

    Good for the head (tiny GRU MatMuls) and for a quick backbone size win on x86.
    ``per_channel`` is off by default (irrelevant for the head's small matrices;
    turn it on for the backbone if you want slightly better conv weights).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as d:
        prepared = Path(d) / "prepared.onnx"
        _prepare(src, prepared)
        quantize_dynamic(str(prepared), str(out), weight_type=QuantType.QInt8, per_channel=per_channel)
    _assert_single_file(out)
    return out


class MelCalibrationReader(CalibrationDataReader):
    """Backbone calibration inputs (log-mel windows) for static quantization.

    Produces them exactly as the runtime does — AGC to −20 dBFS, then the packaged
    ``melspec.onnx``, then the ``WINDOW``-frame window advanced ``MEL_FRAMES_PER_FRAME``
    per hop — so the calibration activations sit on the same manifold the backbone
    sees at inference. Off-manifold calibration (e.g. random mels) widens the MinMax
    ranges and measurably degrades the int8 embedding, so feed real, representative
    −20 dBFS speech (the same positives/backgrounds used to train the heads).
    """

    def __init__(self, wavs_dir: str | Path, input_name: str, *, max_files: int = 50, max_windows: int = 2000) -> None:
        paths = sorted(Path(wavs_dir).rglob("*.wav"))[:max_files]
        if not paths:
            raise RuntimeError(f"no calibration wavs under {wavs_dir}")
        mel_sess = ort.InferenceSession(_packaged("melspec.onnx"), providers=["CPUExecutionProvider"])
        mel_in = mel_sess.get_inputs()[0].name
        agc_win = int(AGC_WINDOW_S * SAMPLE_RATE)
        hop = MEL_FRAMES_PER_FRAME
        windows: list[np.ndarray] = []
        for p in paths:
            audio = _RmsNormalizer(agc_win)(load_wav(p))  # fresh AGC per clip (causal; no cross-file state)
            mel = cast(np.ndarray, mel_sess.run(None, {mel_in: audio[None].astype(np.float32)})[0])[0]  # [T,n_mels]
            for k in range((len(mel) - WINDOW) // hop + 1):
                windows.append(mel[k * hop : k * hop + WINDOW][None].astype(np.float32))
                if len(windows) >= max_windows:
                    break
            if len(windows) >= max_windows:
                break
        if not windows:
            raise RuntimeError(f"no calibration windows from {wavs_dir} (clips shorter than {WINDOW} mel frames?)")
        self._data = iter([{input_name: w} for w in windows])

    def get_next(self) -> dict | None:  # ty: ignore[invalid-method-override] (None is the end-of-data sentinel)
        return next(self._data, None)


def quantize_static_model(
    src: str | Path,
    out: str | Path,
    calib_wavs: str | Path,
    *,
    per_channel: bool = True,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
) -> Path:
    """Static QDQ int8 quantization with a mel calibration set. Returns ``out``.

    QDQ format with int8 weights *and* activations — what the XNNPACK EP consumes on
    ARM64. Only ``Conv`` and ``Gemm`` are quantized, so the loudness z-score and the
    final L2-normalize (the embedding's normalization-critical ops) stay float.
    Sweep ``calibrate_method`` (``Entropy``/``Percentile``) if MinMax regresses the gate.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as d:
        prepared = Path(d) / "prepared.onnx"
        _prepare(src, prepared)
        input_name = onnx.load(str(prepared)).graph.input[0].name
        reader = MelCalibrationReader(calib_wavs, input_name)
        quantize_static(
            str(prepared),
            str(out),
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "Gemm"],
            calibrate_method=calibrate_method,
        )
    _assert_single_file(out)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="int8-quantize an ONNX artifact (Phase 6).")
    p.add_argument("src", help="input .onnx (backbone.onnx or <word>_head.onnx; do NOT quantize melspec.onnx)")
    p.add_argument("out", help="output quantized .onnx")
    p.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    p.add_argument("--calib-wavs", help="dir of representative -20 dBFS wavs (required for --mode static)")
    p.add_argument("--per-channel", action="store_true", help="per-channel weight quantization")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "static":
        if not args.calib_wavs:
            raise SystemExit("--mode static requires --calib-wavs <dir>")
        out = quantize_static_model(args.src, args.out, args.calib_wavs, per_channel=args.per_channel)
    else:
        out = quantize_dynamic_model(args.src, args.out, per_channel=args.per_channel)
    before, after = Path(args.src).stat().st_size, out.stat().st_size
    print(f"{args.mode}: {args.src} {before} B -> {out} {after} B ({100 * (1 - after / before):.0f}% smaller)")
    print("re-gate before shipping: train_head --backbone <int8> (FA/FR) and/or python -m wwd_i.export.compare")


if __name__ == "__main__":
    main()
