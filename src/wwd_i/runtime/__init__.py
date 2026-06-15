"""Streaming inference: load an ONNX model and run frames through it in real time."""

from wwd_i.runtime.harness import StreamStats, load_session, run_stream

__all__ = ["StreamStats", "load_session", "run_stream"]
