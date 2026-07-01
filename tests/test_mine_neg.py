"""Hard-negative mining tests (torch-free).

Reuses the synthetic toy head from ``test_engine``: an always-fire head over synthetic
media yields ``[HEAD_CONTEXT_HOPS, D]`` windows; a never-fire head mines nothing (raises).
"""

import numpy as np
import pytest
import soundfile as sf
from test_engine import _toy_head

from wwd_i.config import SAMPLE_RATE
from wwd_i.runtime.engine import HEAD_CONTEXT_HOPS
from wwd_i.train.mine_neg import build_parser, mine


def _media(d, n, seconds=4):
    d.mkdir()
    for i in range(n):
        sig = np.random.default_rng(i).standard_normal(int(seconds * SAMPLE_RATE)) * 0.1
        sf.write(d / f"m{i}.wav", sig.astype(np.float32), SAMPLE_RATE)


def test_mine_harvests_firing_windows(tmp_path):
    media = tmp_path / "media"
    _media(media, 2)
    head = _toy_head(tmp_path / "always.onnx", bias=20.0)  # fires on any input
    out = tmp_path / "mined.npy"
    args = build_parser().parse_args(
        ["--head", str(head), "--media", str(media), "--out", str(out), "--mine-thr", "0.3", "--max-per-file", "5"]
    )

    emb = mine(args)

    assert emb.ndim == 3 and emb.shape[1:] == (HEAD_CONTEXT_HOPS, 96)  # bg_neg-compatible cache shape
    assert 0 < emb.shape[0] <= 2 * 5  # capped at --max-per-file per file
    assert out.exists() and np.load(out).shape == emb.shape


def test_mine_finds_nothing_when_head_never_fires(tmp_path):
    media = tmp_path / "media"
    _media(media, 1)
    head = _toy_head(tmp_path / "never.onnx", bias=-20.0)  # P(wake) ~ 0
    args = build_parser().parse_args(["--head", str(head), "--media", str(media), "--out", str(tmp_path / "x.npy")])
    with pytest.raises(RuntimeError, match="mine-thr"):
        mine(args)
