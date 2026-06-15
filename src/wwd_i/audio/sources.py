"""Frame sources: a uniform iterator of 80 ms float32 frames from a file or mic."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from wwd_i.audio.io import load_wav, to_frames
from wwd_i.config import FRAME_SAMPLES, SAMPLE_RATE


def file_frames(path: str | Path, *, pad: bool = False) -> Iterator[np.ndarray]:
    """Stream 80 ms frames from an audio file (resampled to SAMPLE_RATE)."""
    yield from to_frames(load_wav(path), pad=pad)


def mic_frames(device: int | str | None = None) -> Iterator[np.ndarray]:
    """Stream 80 ms frames from the default (or given) input device, indefinitely.

    Requires the optional `sounddevice` dependency (PortAudio); install it with
    `uv add sounddevice`. A callback fills a queue that decouples the audio
    thread from the consumer — this queue is the streaming buffer.
    """
    try:
        import sounddevice as sd
    except ModuleNotFoundError as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "microphone capture needs the optional 'sounddevice' package; run `uv add sounddevice`"
        ) from e

    import queue

    q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
        if status:  # pragma: no cover - device under/overflow
            print(f"audio status: {status}")
        q.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        device=device,
        callback=_callback,
    ):
        while True:
            yield q.get()
