import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # preprocess_bg imports train_head, which imports torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.train.preprocess_bg import build_parser, preprocess  # noqa: E402


def _run(tmp_path, **overrides):
    out = tmp_path / "bg_neg.npy"
    noise = tmp_path / "noise_pool"
    argv = [
        "--background",
        str(tmp_path / "bg"),
        "--out",
        str(out),
        "--noise-pool-dir",
        str(noise),
        "--crops-per-file",
        "2",
        "--noise-pool-size",
        "4",
        "--n-bg-neg",
        "100",
    ]
    args = build_parser().parse_args(argv)
    for k, v in overrides.items():
        setattr(args, k, v)
    return out, noise, preprocess(args)


def test_preprocess_embeds_and_skips_bad_files(tmp_path):
    bg = tmp_path / "bg"
    bg.mkdir()
    for i in range(3):  # 2 s clips -> 2 crops each = 6 negatives
        sf.write(
            bg / f"ok{i}.wav", np.random.default_rng(i).standard_normal(SAMPLE_RATE * 2).astype(np.float32), SAMPLE_RATE
        )
    sf.write(bg / "short.wav", np.zeros(SAMPLE_RATE // 2, np.float32), SAMPLE_RATE)  # < min_seconds -> skipped
    (bg / "broken.wav").write_bytes(b"not audio")  # undecodable -> skipped

    out, noise, emb = _run(tmp_path)

    assert out.exists()
    saved = np.load(out)
    assert saved.shape == emb.shape == (6, 10, 96)  # only the 3 valid files contributed
    assert np.allclose(np.linalg.norm(saved, axis=-1), 1.0, atol=1e-4)  # backbone L2-normalizes
    pool_wavs = sorted(noise.glob("*.wav"))
    assert len(pool_wavs) == 4  # capped at --noise-pool-size
    assert all(len(sf.read(p)[0]) == int(1.5 * SAMPLE_RATE) for p in pool_wavs)


def test_preprocess_caps_at_n_bg_neg(tmp_path):
    bg = tmp_path / "bg"
    bg.mkdir()
    for i in range(3):
        sf.write(
            bg / f"ok{i}.wav", np.random.default_rng(i).standard_normal(SAMPLE_RATE * 2).astype(np.float32), SAMPLE_RATE
        )
    _, _, emb = _run(tmp_path, n_bg_neg=3)
    assert emb.shape[0] == 3  # truncated to the requested count


def test_preprocess_normalizes_crops_before_embed(tmp_path, monkeypatch):
    # Every crop must reach the backbone at -20 dBFS (the engine's inference AGC target); a
    # natural-loudness cache trains the head off the served manifold. Capture the buffers handed
    # to embed_clips (both a quiet and a loud source) and assert their RMS hit the target.
    import wwd_i.train.preprocess_bg as pb
    from wwd_i.runtime.engine import AGC_TARGET_RMS

    captured: list[np.ndarray] = []

    def _fake_embed(buf, session, **kw):
        captured.extend(buf)
        return np.zeros((len(buf), 10, 96), np.float32)

    monkeypatch.setattr(pb, "embed_clips", _fake_embed)

    bg = tmp_path / "bg"
    bg.mkdir()
    for i, level in enumerate((0.005, 0.5)):  # quiet + loud sources -> both must normalize to target
        sig = np.random.default_rng(i).standard_normal(SAMPLE_RATE * 2) * level
        sf.write(bg / f"ok{i}.wav", sig.astype(np.float32), SAMPLE_RATE)
    _run(tmp_path)

    rms = [float(np.sqrt(np.mean(np.square(c, dtype=np.float64)))) for c in captured]
    assert captured and all(abs(r - AGC_TARGET_RMS) < 5e-3 for r in rms), rms


def test_preprocess_aug_frac_keeps_shape_and_clean_noise_pool(tmp_path):
    bg = tmp_path / "bg"
    bg.mkdir()
    for i in range(4):
        sf.write(
            bg / f"ok{i}.wav", np.random.default_rng(i).standard_normal(SAMPLE_RATE * 2).astype(np.float32), SAMPLE_RATE
        )
    out, noise, emb = _run(tmp_path, aug_frac=1.0, aug_pool=4)  # augment every crop

    assert emb.shape == (8, 10, 96)  # contract shape unchanged by augmentation (4 files x 2 crops)
    assert np.allclose(np.linalg.norm(emb, axis=-1), 1.0, atol=1e-4)  # backbone still L2-normalizes
    pool_wavs = sorted(noise.glob("*.wav"))
    assert len(pool_wavs) == 4 and all(len(sf.read(p)[0]) == int(1.5 * SAMPLE_RATE) for p in pool_wavs)
