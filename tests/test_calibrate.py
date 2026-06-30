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

    assert set(res) == {"chosen", "det", "neg_hours", "n_impostors"}
    assert res["n_impostors"] == 0  # none passed -> no impostor_far column
    assert all("impostor_far" not in r for r in res["det"])
    assert abs(res["neg_hours"] - 10 / 3600) < 1e-6  # FA/hr denominator is the true audio duration
    fa = [r["fa_per_hr"] for r in res["det"]]
    assert all(fa[i] >= fa[i + 1] for i in range(len(fa) - 1))  # FA non-increasing as threshold rises
    assert max(fa) > 0  # the always-firing head does cross on the negative stream
    # refractory of 1 s caps fires over a 10 s stream to ~10, not one-per-hop (~116)
    assert max(r["fa"] for r in res["det"]) <= 12


def test_calibrate_stream_reports_rhythm_impostor_far(tmp_path):
    """Held-out rhythm impostors get a per-threshold false-accept rate (not folded
    into FA/hr), and target_impostor_far can gate the PASS verdict."""
    pytest.importorskip("torch")
    from test_engine import _toy_head

    from wwd_i.runtime.engine import WakeWordEngine
    from wwd_i.train.train_head import calibrate_stream

    head = _toy_head(tmp_path / "head.onnx")  # fires high on any input
    rng = np.random.default_rng(0)
    for name, n in (("neg", SAMPLE_RATE * 10), ("pos", int(SAMPLE_RATE * 1.5)), ("imp", int(SAMPLE_RATE * 1.5))):
        sf.write(str(tmp_path / f"{name}.wav"), (rng.standard_normal(n) * 0.1).astype(np.float32), SAMPLE_RATE)
    neg, pos, imp = (tmp_path / f"{x}.wav" for x in ("neg", "pos", "imp"))

    eng = WakeWordEngine(head, threshold=0.0, refractory_s=0.0)
    res = calibrate_stream(eng, [neg], [pos], target_fa=0.5, target_fr=0.05, refractory=1.0, impostor_paths=[imp])

    assert res["n_impostors"] == 1
    assert all("impostor_far" in r and 0.0 <= r["impostor_far"] <= 1.0 for r in res["det"])
    assert res["det"][0]["impostor_far"] == 1.0  # always-firing head fires on the impostor at low thr

    # an impossible impostor target can never PASS, even where FA/FR would
    gated = calibrate_stream(
        eng, [neg], [pos], target_fa=0.5, target_fr=0.05, refractory=1.0, impostor_paths=[imp], target_impostor_far=0.0
    )
    assert gated["chosen"]["passed"] is False


def test_calibrate_stream_skips_corrupt_negatives(tmp_path):
    """A corrupt/truncated negative (interrupted download) must be skipped, not crash the
    whole run, and must drop out of the FA/hr duration; all-corrupt is a hard error."""
    pytest.importorskip("torch")
    from test_engine import _toy_head

    from wwd_i.runtime.engine import WakeWordEngine
    from wwd_i.train.train_head import calibrate_stream

    head = _toy_head(tmp_path / "head.onnx")
    rng = np.random.default_rng(0)
    good = tmp_path / "good.wav"
    sf.write(str(good), (rng.standard_normal(SAMPLE_RATE * 10) * 0.1).astype(np.float32), SAMPLE_RATE)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00not-a-real-wav")  # libsndfile fails to read this
    pos = tmp_path / "pos.wav"
    sf.write(str(pos), (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.1).astype(np.float32), SAMPLE_RATE)

    eng = WakeWordEngine(head, threshold=0.0, refractory_s=0.0)
    res = calibrate_stream(eng, [good, bad], [pos], target_fa=0.5, target_fr=0.05, refractory=1.0)
    assert abs(res["neg_hours"] - 10 / 3600) < 1e-6  # only the good file counts toward duration

    with pytest.raises(RuntimeError, match="unreadable"):
        calibrate_stream(eng, [bad], [pos], target_fa=0.5, target_fr=0.05, refractory=1.0)


def test_calib_only_recalibrates_existing_head_without_training(tmp_path):
    """--calib-only re-runs the streaming calibration on an already-exported head and rewrites
    the JSON, doing no training — the namespace carries only calibration args (no n_aug/hidden/
    epochs etc.), so reaching the training path at all would AttributeError."""
    import argparse
    import json

    pytest.importorskip("torch")
    from test_engine import _toy_head

    from wwd_i.train.train_head import train

    head = _toy_head(tmp_path / "w_head.onnx")  # stands in for a previously-trained head
    rng = np.random.default_rng(0)
    calib = tmp_path / "calib_bg"
    calib.mkdir()
    sf.write(str(calib / "neg.wav"), (rng.standard_normal(SAMPLE_RATE * 10) * 0.1).astype(np.float32), SAMPLE_RATE)
    pos = tmp_path / "pos"
    pos.mkdir()
    sf.write(str(pos / "p.wav"), (rng.standard_normal(int(SAMPLE_RATE * 1.5)) * 0.1).astype(np.float32), SAMPLE_RATE)
    out_json = tmp_path / "w_head.json"

    base = {
        "calib_only": True,
        "calib_bg": [str(calib)],
        "positives": str(pos),
        "out": str(head),
        "threshold_out": str(out_json),
        "word": "w",
        "backbone": "",
        "rhythm_impostors": None,
        "target_fa": 0.5,
        "target_fr": 0.05,
        "refractory": 1.0,
        "target_impostor_far": None,
    }
    train(argparse.Namespace(**base))
    meta = json.loads(out_json.read_text())
    assert meta["word"] == "w"
    assert {"threshold", "refractory_seconds", "fa_per_hr", "fr", "passed"} <= set(meta)

    with pytest.raises(RuntimeError, match="needs --calib-bg"):
        train(argparse.Namespace(**{**base, "calib_bg": None}))
    with pytest.raises(RuntimeError, match="no head"):
        train(argparse.Namespace(**{**base, "out": str(tmp_path / "missing_head.onnx")}))
