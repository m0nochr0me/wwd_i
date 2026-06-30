import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("torch")  # train_head imports torch

from wwd_i.config import SAMPLE_RATE  # noqa: E402
from wwd_i.data.augment import Augmenter  # noqa: E402
from wwd_i.train.train_head import AUG_JITTER_SECONDS, _augmented  # noqa: E402


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
