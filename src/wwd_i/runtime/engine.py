"""Streaming wake-word engine: mel -> backbone -> head -> detection events (Phase 5).

Wires the three frozen ONNX stages into a single always-on detector. Audio is
pushed in arbitrary-sized chunks; the engine keeps rolling buffers so the output
is identical whether the same audio arrives as one block or as a stream of 80 ms
frames (the Phase-5 parity gate).

Cadence (must match ``train.train_head._windows``): the log-mel front-end emits
8 mel frames per 80 ms (``MEL_FRAMES_PER_FRAME``); the backbone embeds a rolling
``WINDOW``-frame window advanced 8 frames per hop -> one 96-d embedding every
80 ms; the GRU head consumes one embedding per hop carrying its hidden state and
emits ``P(wake)``; crossing the calibrated threshold fires a detection, then a
refractory window debounces repeats.

Torch-free: every stage runs in ONNX Runtime, so this module (and the whole
inference install) never imports torch.
"""

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import onnxruntime as ort

from wwd_i.audio.io import load_wav
from wwd_i.config import HOP_LENGTH, MEL_FRAMES_PER_FRAME, SAMPLE_RATE
from wwd_i.features.streaming import MelStreamer
from wwd_i.runtime.harness import load_session

# Mel frames per backbone window (~760 ms). MUST equal train.train_head.WINDOW,
# which defines the window the head was trained on; tests/test_engine.py asserts it.
WINDOW = 76

# Loudness normalization (AGC). The frozen backbone is strongly loudness-sensitive
# — a -12 dB input level drop already halves the embedding's cosine similarity, and
# heads are trained/calibrated on -20 dBFS audio (data.elevenlabs._rms_normalize) — so
# raw, quiet mic input lands off the trained manifold and the head's output is arbitrary.
# This pulls the input back toward the training level before the mel front-end.
AGC_TARGET_RMS = 0.1  # -20 dBFS, matches the head-training normalization target
AGC_WINDOW_S = 1.5  # rolling-RMS span, matches train.train_head.CLIP_SECONDS
AGC_MAX_GAIN = 20.0  # +26 dB cap: don't amplify silence/room tone up to speech level


class _RmsNormalizer:
    """Causal moving-RMS automatic gain control toward ``AGC_TARGET_RMS``.

    The gain applied to each sample is a function only of the trailing ``window``
    samples on the absolute audio timeline, so it is identical whether audio
    arrives in one block or as a stream of frames — preserving the streamed==batch
    parity gate. Gain is capped at ``max_gain`` so near-silence is not blown up to
    speech level (which would re-introduce false fires).
    """

    def __init__(self, window: int, target: float = AGC_TARGET_RMS, max_gain: float = AGC_MAX_GAIN) -> None:
        self._win = max(1, window)
        self._target = target
        self._max_gain = max_gain
        self._tail = np.zeros(0, dtype=np.float64)  # trailing squared samples (≤ window-1) carried across pushes

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32)
        sq = x.astype(np.float64) ** 2
        buf = np.concatenate([self._tail, sq])
        prefix = np.concatenate([[0.0], np.cumsum(buf)])  # prefix[k] = sum(buf[:k])
        hi = np.arange(self._tail.size + 1, self._tail.size + 1 + x.size)  # window end (exclusive prefix index)
        lo = np.maximum(0, hi - self._win)
        rms = np.sqrt((prefix[hi] - prefix[lo]) / (hi - lo))
        gain = np.minimum(self._target / np.maximum(rms, 1e-9), self._max_gain)
        self._tail = buf[-(self._win - 1) :] if self._win > 1 else buf[:0]
        return (x.astype(np.float64) * gain).astype(np.float32)


def _low_cpu_options() -> ort.SessionOptions:
    """Session options for the always-on detector: single-threaded, no spinning.

    By default ORT spawns one intra-op worker per core, each busy-waiting between
    ``Run`` calls. At ~12.5 hops/s across three tiny models those pools spin
    through the 80 ms idle gaps and peg multiple cores for almost no real work
    (measured ~7 cores). One thread with spinning disabled does the ~1 ms/hop
    compute on the calling thread instead — CPU drops to a fraction of one core
    with no latency cost at this model size.
    """
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return o


@dataclass(frozen=True)
class Detection:
    """A wake-word firing: ``time_s`` is the audio time at the end of the window
    that triggered it; ``score`` is ``P(wake)`` at that hop."""

    time_s: float
    score: float


def _packaged(name: str) -> str:
    """Path to an ONNX artifact shipped inside ``wwd_i.models``."""
    return str(files("wwd_i.models") / name)


class WakeWordEngine:
    """Stateful streaming detector for one wake word.

    Feed audio with :meth:`push`; it returns the detections that fired during that
    call. Construct one engine per stream and :meth:`reset` between independent
    streams (or use :func:`detect_file`).
    """

    def __init__(
        self,
        head_path: str | Path,
        *,
        calibration: str | Path | None = None,
        backbone_path: str | Path = "",
        mel_path: str | Path = "",
        threshold: float | None = None,
        refractory_s: float | None = None,
        normalize: bool = True,
    ) -> None:
        head_path = Path(head_path)
        meta = self._load_calibration(head_path, calibration)
        self.word: str = meta.get("word", head_path.stem)
        if threshold is None:
            threshold = meta.get("threshold")
        if threshold is None:
            raise ValueError(
                f"no threshold: pass threshold=... or a calibration json (looked for {head_path.with_suffix('.json')})"
            )
        self.threshold = float(threshold)
        self.refractory_s = float(refractory_s if refractory_s is not None else meta.get("refractory_seconds", 1.0))

        opts = _low_cpu_options()
        self._mel_session = load_session(mel_path or _packaged("melspec.onnx"), opts)
        self._backbone = load_session(backbone_path or _packaged("backbone.onnx"), opts)
        self._head = load_session(str(head_path), opts)

        self._mel_in = self._mel_session.get_inputs()[0].name
        self._bb_in = self._backbone.get_inputs()[0].name
        self._emb_in, self._h_in = (i.name for i in self._head.get_inputs())
        self._prob_out, self._hn_out = (o.name for o in self._head.get_outputs())
        self._hidden = int(self._head.get_inputs()[1].shape[-1])
        self._normalize = normalize

        self.reset()

    @staticmethod
    def _load_calibration(head_path: Path, calibration: str | Path | None) -> dict:
        path = Path(calibration) if calibration is not None else head_path.with_suffix(".json")
        if path.exists():
            return json.loads(path.read_text())
        if calibration is not None:  # explicitly asked for a file that isn't there
            raise FileNotFoundError(f"calibration json not found: {path}")
        return {}

    def reset(self) -> None:
        """Clear all streaming state — buffers, GRU hidden state, timers."""
        self._mel = MelStreamer(self._mel_fn)
        self._agc = _RmsNormalizer(int(AGC_WINDOW_S * SAMPLE_RATE)) if self._normalize else None
        self._mel_buf = np.zeros((0, self._n_mels()), dtype=np.float32)
        self._h = np.zeros((1, 1, self._hidden), dtype=np.float32)
        self._hop = 0
        self._last_fire_s = -math.inf

    def _n_mels(self) -> int:
        return int(self._backbone.get_inputs()[0].shape[-1])

    def _mel_fn(self, signal: np.ndarray) -> np.ndarray:
        """Run the mel ONNX over a 1-D signal -> ``[frames, n_mels]`` (MelStreamer adapter)."""
        out = self._mel_session.run(None, {self._mel_in: signal[None].astype(np.float32)})[0]
        return out[0]

    def push(self, samples: np.ndarray) -> list[Detection]:
        """Feed an audio chunk (any length, 16 kHz mono float32); return detections."""
        samples = np.asarray(samples, dtype=np.float32)
        if self._agc is not None:
            samples = self._agc(samples)
        new = self._mel.push(samples)
        if new.shape[0]:
            self._mel_buf = np.concatenate([self._mel_buf, new], axis=0)

        dets: list[Detection] = []
        while self._mel_buf.shape[0] >= WINDOW:
            window = self._mel_buf[:WINDOW][None]  # [1, WINDOW, n_mels]
            emb = self._backbone.run(None, {self._bb_in: window})[0]  # [1, D]
            prob, self._h = self._head.run(
                [self._prob_out, self._hn_out],
                {self._emb_in: emb[None].astype(np.float32), self._h_in: self._h},
            )
            t_end = (self._hop * MEL_FRAMES_PER_FRAME + WINDOW) * HOP_LENGTH / SAMPLE_RATE
            self._hop += 1
            self._mel_buf = self._mel_buf[MEL_FRAMES_PER_FRAME:]

            score = float(np.asarray(prob).reshape(-1)[0])
            if score >= self.threshold and (t_end - self._last_fire_s) >= self.refractory_s:
                dets.append(Detection(time_s=t_end, score=score))
                self._last_fire_s = t_end
        return dets


def detect_file(engine: WakeWordEngine, path: str | Path) -> list[Detection]:
    """Offline detection over a whole file (the batch reference for the parity gate)."""
    engine.reset()
    return engine.push(load_wav(path))
