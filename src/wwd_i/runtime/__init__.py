"""Streaming inference: load an ONNX model and run frames through it in real time."""

from wwd_i.runtime.engine import Detection, WakeWordEngine, detect_file
from wwd_i.runtime.harness import StreamStats, load_session, run_stream

__all__ = [
    "Detection",
    "StreamStats",
    "WakeWordEngine",
    "detect_file",
    "load_session",
    "run_stream",
]
