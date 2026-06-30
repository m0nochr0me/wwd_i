"""Structural quality gate for TTS wake-word positives (torch-free).

A usable positive is a *single voiced burst* that sits fully inside the head's
front-aligned ``CLIP_SECONDS`` crop (``train_head.CLIP_SECONDS``) with a prompt
onset. TTS backends occasionally render clips that violate this and silently
poison a head:

- ``multi-burst`` — a disfluent / repeated take ("Aksana, hmmm... Axana?"): two
  or more voiced spans separated by a real pause.
- ``truncated`` — speech is still going at the crop boundary, so the front crop
  cuts the word.
- ``late-onset`` — the word starts so late that after the crop the positive is
  near-silent → teaches "silence ramp ⇒ fire" (a false-accept).
- ``near-silent`` — effectively empty render.

Detection is energy-VAD only (numpy, no ASR): it catches the *structural* faults
above but not a clean single-burst *mis*pronunciation — that needs transcription
and is out of scope here. Shared by ``generate_clips`` (reject at write time) and
the notebook's positives-sanitation step.
"""

import numpy as np

from wwd_i.config import SAMPLE_RATE

# Must equal ``train_head.CLIP_SECONDS`` — the front-aligned crop the head trains
# on. Mirrored (not imported) to keep this module torch-free; the equality is
# pinned by ``tests/test_quality.py::test_clip_seconds_matches_trainer``.
CLIP_SECONDS = 1.5

MAX_ONSET_SECONDS = 0.9  # word must begin within this, else it lands near/over the crop edge
MIN_GAP_SECONDS = 0.25  # silence longer than this between voiced frames splits a burst
VAD_REL_DB = -25.0  # a frame is "voiced" if its RMS is within this many dB of the loudest frame
MIN_PEAK = 0.02  # below this absolute peak the render is treated as silence
MIN_ACTIVE_FRAMES = 3  # fewer voiced frames than this ⇒ no usable speech

_FRAME = 320  # 20 ms VAD analysis frame @ 16 kHz
_HOP = 160  # 10 ms VAD hop


def clip_defects(wav: np.ndarray) -> list[str]:
    """Return structural defect tags for a 16 kHz mono positive; empty ⇒ clean.

    Tags: ``near-silent``, ``late-onset``, ``truncated``, ``multi-burst`` (a clip
    may earn several). ``wav`` is a 1-D float waveform at ``SAMPLE_RATE``.
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0

    n = (len(wav) - _FRAME) // _HOP + 1
    if peak < MIN_PEAK or n <= 0:
        return ["near-silent"]
    energy = np.sqrt(
        np.array([np.mean(wav[i * _HOP : i * _HOP + _FRAME] ** 2) for i in range(n)], dtype=np.float64) + 1e-12
    )
    thr = energy.max() * 10.0 ** (VAD_REL_DB / 20.0)
    active = np.flatnonzero(energy > thr)
    if len(active) < MIN_ACTIVE_FRAMES:
        return ["near-silent"]

    onset = active[0] * _HOP / SAMPLE_RATE
    offset = (active[-1] * _HOP + _FRAME) / SAMPLE_RATE
    bursts = 1 + int(np.count_nonzero(np.diff(active) > MIN_GAP_SECONDS * SAMPLE_RATE / _HOP))

    defects: list[str] = []
    if onset > MAX_ONSET_SECONDS:
        defects.append("late-onset")
    if offset > CLIP_SECONDS:
        defects.append("truncated")
    if bursts >= 2:
        defects.append("multi-burst")
    return defects
