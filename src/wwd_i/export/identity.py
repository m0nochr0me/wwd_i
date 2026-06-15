"""Build a trivial identity ONNX model that returns its input unchanged.

Used to verify the streaming ONNX-Runtime harness end-to-end before any real
model exists (Phase 0)."""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from wwd_i.config import FRAME_SAMPLES

INPUT_NAME = "frame"
OUTPUT_NAME = "out"


def build_identity_model(path: str | Path, length: int = FRAME_SAMPLES) -> Path:
    """Write an ONNX model mapping a float32 [length] tensor to itself."""
    inp = helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [length])
    out = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [length])
    node = helper.make_node("Identity", [INPUT_NAME], [OUTPUT_NAME])
    graph = helper.make_graph([node], "identity", [inp], [out])
    model = helper.make_model(graph, producer_name="wwd_i")
    onnx.checker.check_model(model)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
