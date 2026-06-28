from typing import cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wwd_i.config import N_MELS, SAMPLE_RATE  # noqa: E402
from wwd_i.models.head import HeadConfig, WakeHead, export_head_onnx  # noqa: E402
from wwd_i.train.train_head import WINDOW, _default_backbone, _windows, calibrate, embed_clips  # noqa: E402

D = 96


def test_head_shapes_and_clip_score():
    head = WakeHead().eval()
    emb = torch.randn(2, 10, D)
    logits, hn = head(emb)
    assert logits.shape == (2, 10) and hn.shape == (1, 2, 48)
    assert head.clip_logits(emb).shape == (2,)
    s = head.clip_score(emb, 3)  # top-k-mean probability per clip
    assert s.shape == (2,) and bool(((s >= 0) & (s <= 1)).all())


def test_head_trains_on_separable_data():
    from wwd_i.train.train_head import HEAD_TOPK

    g = torch.Generator().manual_seed(0)
    pos = torch.randn(64, 10, D, generator=g) * 0.1
    pos[:, 3:8, 0] += 3.0  # a SUSTAINED response over several hops (the top-k criterion, not a lone spike)
    neg = torch.randn(64, 10, D, generator=g) * 0.1
    x = torch.cat([pos, neg])
    y = torch.cat([torch.ones(64), torch.zeros(64)])

    head = WakeHead(HeadConfig(hidden=32))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    lossfn = torch.nn.BCELoss()  # clip_score is a probability
    for _ in range(200):
        loss = lossfn(head.clip_score(x, HEAD_TOPK), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    acc = ((head.clip_score(x, HEAD_TOPK) >= 0.5).float() == y).float().mean()
    assert acc > 0.9


def test_head_onnx_streaming_parity(tmp_path):
    import onnxruntime as ort

    head = WakeHead().eval()
    path = export_head_onnx(head, tmp_path / "head.onnx")
    assert not path.with_suffix(".onnx.data").exists()  # self-contained
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    emb = torch.randn(1, 12, D)
    with torch.no_grad():
        ref = torch.sigmoid(head(emb)[0]).numpy()[0]  # [12]
    h = np.zeros((1, 1, 48), dtype=np.float32)
    probs = []
    for t in range(12):  # advance one hop at a time, carrying state
        p, h = sess.run(None, {"embedding": emb[:, t : t + 1, :].numpy(), "h0": h})
        probs.append(float(cast(np.ndarray, p)[0, 0]))
    assert np.max(np.abs(np.array(probs) - ref)) < 1e-3  # streaming == batch


def test_windows_shape():
    w = _windows(np.zeros((148, N_MELS), dtype=np.float32))  # 1.5 s clip -> 10 hops
    assert w.shape == (10, WINDOW, N_MELS)


def test_embed_clips_with_packaged_backbone():
    import onnxruntime as ort

    sess = ort.InferenceSession(_default_backbone(), providers=["CPUExecutionProvider"])
    clips = [np.random.default_rng(i).standard_normal(int(1.5 * SAMPLE_RATE)).astype(np.float32) for i in range(5)]
    emb = embed_clips(clips, sess, batch=2)  # exercises the chunk boundary
    assert emb.shape == (5, 10, D)
    assert np.allclose(np.linalg.norm(emb, axis=-1), 1.0, atol=1e-4)  # backbone L2-normalizes
    assert np.allclose(emb, embed_clips(clips, sess, batch=100), atol=1e-5)  # chunked == single call


def test_calibrate_pass_and_fail():
    y = np.concatenate([np.ones(50), np.zeros(200)])
    ok = calibrate(np.concatenate([np.full(50, 0.9), np.full(200, 0.05)]), y, target_fa=0.5, target_fr=0.05)
    assert ok["chosen"]["passed"]

    yb = np.concatenate([np.ones(50), np.zeros(4000)])
    bad = calibrate(np.concatenate([np.full(50, 0.4), np.full(4000, 0.6)]), yb, target_fa=0.5, target_fr=0.05)
    assert not bad["chosen"]["passed"]  # positives below negatives -> no good operating point
