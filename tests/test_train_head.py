import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # train_head imports torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.augment import Augmenter  # noqa: E402
from wwd_i.runtime.engine import AGC_TARGET_RMS  # noqa: E402
from wwd_i.train.train_head import AUG_JITTER_SECONDS, _augmented  # noqa: E402


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def _leading_zeros(x: np.ndarray) -> int:
    nz = np.flatnonzero(x)
    return int(nz[0]) if len(nz) else len(x)


def test_augmented_jitters_onset_but_not_the_anchor(tmp_path):
    length = int(1.5 * SAMPLE_RATE)
    wav = tmp_path / "pos.wav"
    sf.write(wav, np.full(length, 0.5, np.float32), SAMPLE_RATE)  # constant -> leading zeros == front-pad

    n_aug = 8
    ident = Augmenter(None, p_gain=0, p_speed=0, p_pitch=0, p_rir=0, p_noise=0, p_clip=0, seed=0)  # identity aug
    out = _augmented([wav], ident, n_aug, length)

    assert len(out) == 1 + n_aug
    assert all(len(c) == length for c in out)  # every clip re-fixed to length
    assert _leading_zeros(out[0]) == 0  # the clean anchor is NOT jittered

    pads = [_leading_zeros(c) for c in out[1:]]
    max_pad = int(AUG_JITTER_SECONDS * SAMPLE_RATE)
    assert max(pads) > 0  # onset slid on at least one variant (seeded -> deterministic)
    assert all(0 <= p <= max_pad for p in pads)  # jitter stays bounded


def test_augmented_normalizes_to_agc_target(tmp_path):
    # Training clips must reach the backbone at the -20 dBFS the engine's AGC produces at inference;
    # the loudness-sensitive backbone otherwise embeds them off the served manifold (persistent
    # media FA that more negatives can't fix). Inputs are deliberately off-level (very quiet / loud)
    # -> every emitted clip (clean anchor and post-aug variants) must come back at AGC_TARGET_RMS.
    length = int(1.5 * SAMPLE_RATE)
    for i, level in enumerate((0.01, 0.6)):
        sf.write(tmp_path / f"p{i}.wav", np.full(length, level, np.float32), SAMPLE_RATE)
    paths = sorted(tmp_path.glob("p*.wav"))
    ident = Augmenter(None, p_gain=0, p_speed=0, p_pitch=0, p_rir=0, p_noise=0, p_clip=0, seed=0)

    out = _augmented(paths, ident, n_aug=2, length=length)

    assert out and all(abs(_rms(c) - AGC_TARGET_RMS) < 1e-3 for c in out), [_rms(c) for c in out]
