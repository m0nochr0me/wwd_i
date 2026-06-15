"""Streaming wrapper for the log-mel front-end.

Wraps a batch log-mel function and feeds it audio incrementally, emitting the mel
frames that become available with each pushed chunk. It keeps a small sample
buffer (the overlap between adjacent frames) so the concatenation of streamed
outputs equals the batch result on the same audio.
"""

from collections.abc import Callable

import numpy as np

from wwd_i.config import HOP_LENGTH, N_MELS, WIN_LENGTH


class MelStreamer:
    """Turn ``mel_fn(signal) -> [frames, n_mels]`` into an incremental feeder."""

    def __init__(self, mel_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._mel_fn = mel_fn
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Append `samples` and return the newly available mel frames.

        Returns an empty ``[0, n_mels]`` array when not enough audio has
        accumulated to form a frame.
        """
        self._buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32)])
        if len(self._buf) < WIN_LENGTH:
            return np.zeros((0, N_MELS), dtype=np.float32)
        out = np.asarray(self._mel_fn(self._buf), dtype=np.float32)
        self._buf = self._buf[out.shape[0] * HOP_LENGTH :]
        return out
