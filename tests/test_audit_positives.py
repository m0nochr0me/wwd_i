"""Positive-audit tests (torch-free).

Reuses the synthetic toy head from ``test_engine`` (no trained artifact, no torch):
a large-negative-bias head keeps P(wake) ~0 so every positive is flagged, exercising
the score -> sort -> flag -> prune path that removes glitched/short renders.
"""

import numpy as np
import soundfile as sf
from test_engine import _toy_head

from wwd_i.config import SAMPLE_RATE
from wwd_i.train.audit_positives import audit, build_parser


def _write_wavs(d, n):
    d.mkdir()
    for i in range(n):
        sig = np.random.default_rng(i).standard_normal(int(1.5 * SAMPLE_RATE)) * 0.1
        sf.write(d / f"p{i}.wav", sig.astype(np.float32), SAMPLE_RATE)


def test_audit_scores_sorts_and_prunes(tmp_path):
    pos = tmp_path / "pos"
    _write_wavs(pos, 3)
    head = _toy_head(tmp_path / "never.onnx", bias=-20.0)  # P(wake) ~ 0 on any input -> all flagged
    csv = tmp_path / "audit.csv"
    args = build_parser().parse_args(
        ["--head", str(head), "--positives", str(pos), "--min-score", "0.3", "--out-csv", str(csv), "--prune"]
    )

    rows = audit(args)

    assert len(rows) == 3
    assert [r[1] for r in rows] == sorted(r[1] for r in rows)  # worst-first
    assert all(s < 0.3 for _, s, _, _ in rows)
    rej = pos.parent / "pos_rejects"
    assert len(list(rej.glob("*.wav"))) == 3 and not list(pos.glob("*.wav"))  # all moved out
    assert csv.exists() and csv.read_text().splitlines()[0] == "path,best_score,n_hops,seconds"


def test_audit_keeps_positives_above_threshold(tmp_path):
    pos = tmp_path / "pos"
    _write_wavs(pos, 2)
    head = _toy_head(tmp_path / "always.onnx", bias=20.0)  # P(wake) ~ 1 -> nothing flagged
    args = build_parser().parse_args(["--head", str(head), "--positives", str(pos), "--min-score", "0.3", "--prune"])

    rows = audit(args)

    assert len(rows) == 2 and all(s > 0.3 for _, s, _, _ in rows)
    assert len(list(pos.glob("*.wav"))) == 2  # none moved
    assert not (pos.parent / "pos_rejects").exists()
