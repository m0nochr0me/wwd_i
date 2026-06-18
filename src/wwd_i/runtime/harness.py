"""ONNX-Runtime streaming harness: feed 80 ms frames through a model, collect
per-frame outputs, and track compute time against the real-time budget."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from wwd_i.config import FRAME_MS


@dataclass(frozen=True)
class StreamStats:
    """Throughput of a streaming run."""

    frames: int
    audio_ms: float
    compute_ms: float

    @property
    def real_time_factor(self) -> float:
        """compute_ms / audio_ms; < 1.0 means faster than real time."""
        return self.compute_ms / self.audio_ms if self.audio_ms else 0.0


def load_session(path: str | Path, sess_options: ort.SessionOptions | None = None) -> ort.InferenceSession:
    """Load an ONNX model on the CPU execution provider."""
    return ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])


def run_stream(
    session: ort.InferenceSession,
    frames: Iterable[np.ndarray],
    on_output: Callable[[int, np.ndarray], None] | None = None,
) -> StreamStats:
    """Run each frame through `session` (single-input model).

    Calls `on_output(index, output)` per frame when provided, and returns
    throughput stats for the whole stream.
    """
    input_name = session.get_inputs()[0].name
    count = 0
    compute = 0.0
    for index, frame in enumerate(frames):
        t0 = perf_counter()
        (out,) = session.run(None, {input_name: frame})
        compute += perf_counter() - t0
        count = index + 1
        if on_output is not None:
            on_output(index, out)
    return StreamStats(frames=count, audio_ms=count * FRAME_MS, compute_ms=compute * 1000.0)
