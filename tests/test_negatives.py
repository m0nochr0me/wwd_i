import numpy as np

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.negatives import GENERIC_CONFUSABLES, hard_negative_phrases, sample_background_clips


def test_hard_negatives_multiword():
    negs = hard_negative_phrases("Hey Computer")
    assert "hey" in negs and "computer" in negs  # sub-words
    assert "hey computer" not in negs  # never the full phrase
    assert all(g in negs for g in GENERIC_CONFUSABLES)


def test_hard_negatives_extra_confusables():
    negs = hard_negative_phrases("Aliyah", extra=["Maria", "All Year", "aliyah", "  "])
    assert "maria" in negs and "all year" in negs  # extra lower-cased and included
    assert "aliyah" not in negs  # extra equal to the wake phrase dropped
    assert all(g in negs for g in GENERIC_CONFUSABLES)  # generics still present


def test_hard_negatives_singleword_no_subwords():
    negs = hard_negative_phrases("marvin")
    assert "marvin" not in negs
    assert "hey" in negs  # generics still present


def test_sample_background_clips_length():
    pool = [np.random.default_rng(0).standard_normal(SAMPLE_RATE * 2).astype(np.float32)]
    clips = sample_background_clips(pool, 5, length=SAMPLE_RATE, seed=0)
    assert len(clips) == 5
    assert all(c.shape == (SAMPLE_RATE,) and c.dtype == np.float32 for c in clips)
