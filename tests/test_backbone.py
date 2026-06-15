import numpy as np
import pytest

pytest.importorskip("torchaudio")  # backbone training stack needs the `train` group

import torch  # noqa: E402

from wwd_i.config import N_MELS, SAMPLE_RATE  # noqa: E402
from wwd_i.data.episodic import EpisodicSampler  # noqa: E402
from wwd_i.features.melspec import MelSpectrogram  # noqa: E402
from wwd_i.models.backbone import Backbone, BackboneConfig, export_onnx  # noqa: E402
from wwd_i.train.probe import few_shot_probe  # noqa: E402
from wwd_i.train.proto import prototypical_loss  # noqa: E402

# Small/fast backbone so the smoke tests stay quick on CPU.
TINY = BackboneConfig(
    stage_channels=(8, 12, 16, 24),
    blocks_per_stage=(1, 1, 1, 1),
    dilations=(1, 2, 4, 8),
    embedding_dim=32,
    dropout=0.0,
)

# Synthetic "words": each class is a distinct fundamental frequency. Separable in
# spectrum, so a trained embedding (and a kNN over it) cluster same-class clips.
CLASS_FREQS = {f"w{i}": 200.0 + 90.0 * i for i in range(12)}
TRAIN_WORDS = [f"w{i}" for i in range(8)]
HELD_WORDS = [f"w{i}" for i in range(8, 12)]


def _word_wave(freq: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    wave = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    wave = wave + 0.05 * rng.standard_normal(SAMPLE_RATE)
    return (0.3 * wave).astype(np.float32)


def _load(label: str, item: int) -> np.ndarray:
    freq = CLASS_FREQS[label] * (1.0 + 0.02 * np.sin(item))  # tiny per-clip jitter
    return _word_wave(freq, seed=item)


def _index(words: list[str], n_per: int = 16) -> dict[str, list[int]]:
    return {w: list(range(n_per)) for w in words}


def test_embedding_shape_and_norm():
    mel = MelSpectrogram().eval()
    model = Backbone(TINY).eval()
    audio = torch.from_numpy(np.stack([_load("w0", i) for i in range(4)]))
    with torch.no_grad():
        emb = model(mel(audio))
    assert emb.shape == (4, TINY.embedding_dim)
    assert torch.allclose(emb.norm(dim=1), torch.ones(4), atol=1e-5)


def test_prototypical_training_converges():
    torch.manual_seed(0)
    mel = MelSpectrogram().eval()
    model = Backbone(TINY)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sampler = EpisodicSampler(_index(TRAIN_WORDS), _load, seed=0)

    first, last = None, []
    for step in range(120):
        model.train()
        audio = sampler.episode(n_way=4, k_shot=3, q_query=3)
        loss, acc = prototypical_loss(model(mel(audio)), 4, 3, 3)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        if step >= 110:
            last.append(acc.item())

    assert last and sum(last) / len(last) > 0.6  # 4-way: chance is 0.25
    assert loss.item() < first  # loss decreased over training


def test_onnx_parity(tmp_path):
    import onnxruntime as ort

    model = Backbone(TINY).eval()
    path = export_onnx(model, tmp_path / "backbone.onnx")
    assert not path.with_suffix(".onnx.data").exists()  # self-contained: weights inline, no sidecar
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for batch, frames in ((1, 76), (1, 98), (5, 76)):  # stream window, full clip, batched embed
        mel = torch.randn(batch, frames, N_MELS)
        with torch.no_grad():
            ref = model(mel).numpy()
        onnx_out = sess.run(None, {"mel": mel.numpy()})[0]
        assert onnx_out.shape == ref.shape == (batch, model.config.embedding_dim)
        assert np.max(np.abs(onnx_out - ref)) < 1e-3


def test_probe_machinery_on_held_out():
    torch.manual_seed(0)
    mel = MelSpectrogram().eval()
    model = Backbone(TINY)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_sampler = EpisodicSampler(_index(TRAIN_WORDS), _load, seed=0)
    held_sampler = EpisodicSampler(_index(HELD_WORDS), _load, seed=1)

    for _ in range(120):
        model.train()
        audio = train_sampler.episode(n_way=4, k_shot=3, q_query=3)
        loss, _ = prototypical_loss(model(mel(audio)), 4, 3, 3)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    def embed(audio):
        return model(mel(audio))

    def baseline(audio):
        return torch.nn.functional.normalize(mel(audio).mean(dim=1), dim=1)

    res = few_shot_probe(held_sampler, embed, baseline, n_way=4, k_shot=3, q_query=3, episodes=20)
    # Machinery check: both classifiers run and the trained embedding generalizes
    # to *unseen* synthetic words. (Backbone-vs-baseline margin is the real gate,
    # and is evaluated on Speech Commands, not on these easily-separable tones.)
    assert 0.0 <= res["baseline"] <= 1.0
    assert res["backbone"] > 0.6  # 4-way held-out, chance 0.25
