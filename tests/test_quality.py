import numpy as np
import pytest

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.quality import CLIP_SECONDS, MAX_ONSET_SECONDS, clip_defects


def _burst(onset_s: float, dur_s: float, amp: float = 0.2) -> np.ndarray:
    """A single voiced span (220 Hz tone) of ``dur_s`` placed at ``onset_s`` in a 4 s buffer."""
    out = np.zeros(int(4.0 * SAMPLE_RATE), dtype=np.float32)
    t = np.linspace(0, dur_s, int(dur_s * SAMPLE_RATE), endpoint=False)
    tone = amp * np.sin(2 * np.pi * 220 * t)
    start = int(onset_s * SAMPLE_RATE)
    out[start : start + len(tone)] = tone.astype(np.float32)
    return out


def test_clean_clip_passes():
    assert clip_defects(_burst(0.1, 0.6)) == []


def test_near_silent_rejected():
    assert clip_defects(np.zeros(SAMPLE_RATE, dtype=np.float32)) == ["near-silent"]
    assert clip_defects(_burst(0.1, 0.6, amp=0.001)) == ["near-silent"]


def test_late_onset_rejected():
    # word starts after MAX_ONSET_SECONDS -> near-silent after the front crop
    assert "late-onset" in clip_defects(_burst(MAX_ONSET_SECONDS + 0.3, 0.4))


def test_truncated_rejected():
    # speech still active at the 1.5 s crop boundary
    assert "truncated" in clip_defects(_burst(0.1, CLIP_SECONDS + 0.5))


def test_multi_burst_rejected():
    # two voiced spans separated by a clear pause = disfluency / repeat
    clip = _burst(0.1, 0.3) + _burst(0.9, 0.3)
    assert "multi-burst" in clip_defects(clip)


def test_clip_seconds_matches_trainer():
    # quality.CLIP_SECONDS is a torch-free mirror of the head's training crop
    pytest.importorskip("torch")
    from wwd_i.train.train_head import CLIP_SECONDS as TRAIN_CLIP_SECONDS

    assert CLIP_SECONDS == TRAIN_CLIP_SECONDS
