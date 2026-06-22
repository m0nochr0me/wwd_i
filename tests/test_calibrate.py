"""Streaming-calibration tests (Phase 4).

``_count_fires`` is the refractory debounce that must mirror ``WakeWordEngine.push``;
``calibrate_stream`` runs the *real* engine over continuous negatives so its FA/hr is
the honest streaming rate. The integration test reuses the synthetic toy head from
``test_engine`` (no trained artifact, no training-side torch needed beyond the import).
"""

import numpy as np
import pytest
import soundfile as sf

from wwd_i.config import SAMPLE_RATE


def test_count_fires_debounces_by_refractory():
    pytest.importorskip("torch")
    from wwd_i.train.train_head import _count_fires

    times = np.array([0.0, 0.5, 1.2, 1.3, 2.4])
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    assert _count_fires(times, scores, 0.5, 1.0) == 3  # 0.0, 1.2, 2.4 — 0.5 & 1.3 debounced
    assert _count_fires(times, scores, 0.5, 0.0) == 5  # no refractory -> every crossing fires
    assert _count_fires(times, scores, 0.95, 1.0) == 0  # nothing crosses
    assert _count_fires(np.array([]), np.array([]), 0.5, 1.0) == 0  # empty stream


def test_calibrate_stream_runs_real_engine_and_sweeps(tmp_path):
    pytest.importorskip("torch")
    from test_engine import _toy_head

    from wwd_i.runtime.engine import WakeWordEngine
    from wwd_i.train.train_head import calibrate_stream

    head = _toy_head(tmp_path / "head.onnx")  # fires high on any input
    rng = np.random.default_rng(0)
    neg = tmp_path / "neg.wav"
    sf.write(str(neg), (rng.standard_normal(SAMPLE_RATE * 10) * 0.1).astype(np.float32), SAMPLE_RATE)
    pos = tmp_path / "pos.wav"
    sf.write(str(pos), (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.1).astype(np.float32), SAMPLE_RATE)

    eng = WakeWordEngine(head, threshold=0.0, refractory_s=0.0)
    res = calibrate_stream(eng, [neg], [pos], target_fa=0.5, target_fr=0.05, refractory=1.0)

    assert set(res) == {"chosen", "det", "neg_hours"}
    assert abs(res["neg_hours"] - 10 / 3600) < 1e-6  # FA/hr denominator is the true audio duration
    fa = [r["fa_per_hr"] for r in res["det"]]
    assert all(fa[i] >= fa[i + 1] for i in range(len(fa) - 1))  # FA non-increasing as threshold rises
    assert max(fa) > 0  # the always-firing head does cross on the negative stream
    # refractory of 1 s caps fires over a 10 s stream to ~10, not one-per-hop (~116)
    assert max(r["fa"] for r in res["det"]) <= 12
