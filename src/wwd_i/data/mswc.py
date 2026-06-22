"""MSWC (Multilingual Spoken Words) scale-up corpus (Phase 2).

Speech Commands proved the pipeline on ~24 training words; MSWC scales the
vocabulary into the hundreds/thousands so the frozen embedding generalizes to
arbitrary wake words. The corpus is streamed from the Hugging Face hub
(``MLCommons/ml_spoken_words``). Because the stream is ordered alphabetically by
word, words are chosen by a hash (uniform across the alphabet, independent of
order) so the kept vocabulary spans ``a``-``z`` rather than just early letters;
the kept clips are materialized as 16 kHz ``{word}/{n}.wav`` under ``root`` and
trained from that directory via the shared ``clips.py`` path.

`datasets` is a training-only dep (the `train` group), imported lazily. Run
``uv run --group train python -m wwd_i.data.mswc --inspect`` to verify the dataset
schema/decoding before a large prepare. See docs/implementation-plan.md Phase 2.
"""

import argparse
import hashlib
import io
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.clips import index_clips, split_samplers
from wwd_i.data.episodic import EpisodicSampler


class _WordSelector:
    """Deterministic, order-independent selection of ~``n_words`` words.

    A word is kept iff ``hash(word) mod max(n_words, vocab_total) < n_words`` — a
    fixed ~``n_words / vocab_total`` fraction, spread uniformly across the alphabet
    regardless of stream ordering. ``vocab_total`` estimates the corpus vocabulary
    size and is used only to hit the target count.
    """

    def __init__(self, n_words: int, vocab_total: int, seed: int = 0) -> None:
        self.n_words = n_words
        self.modulus = max(n_words, vocab_total)
        self.seed = seed

    def keep(self, word: str) -> bool:
        digest = hashlib.blake2b(f"{self.seed}:{word}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.modulus < self.n_words


def _load_stream(dataset: str, config: str, split: str):
    import datasets

    from wwd_i.data._datasets_compat import patch_datasets_py314

    patch_datasets_py314()  # py3.14: fix datasets' _batch_setitems fingerprint override
    # MSWC ships as a loading script; datasets >= 4 dropped script support entirely.
    if int(datasets.__version__.split(".", 1)[0]) >= 4:
        raise RuntimeError(
            f"datasets {datasets.__version__} removed script-based datasets, which {dataset} still uses. "
            "Install an older release:  uv pip install --python .venv/bin/python 'datasets<4'"
        )
    stream = datasets.load_dataset(dataset, config, split=split, streaming=True, trust_remote_code=True)
    # Decode audio ourselves with soundfile; datasets' Audio decoder imports librosa
    # (which needs numba -> no py3.14 wheel), so turn its decoding off.
    return stream.cast_column("audio", datasets.Audio(decode=False))


def _decode_16k_mono(audio: dict) -> np.ndarray:
    """Undecoded HF audio ({bytes|path}) -> 16 kHz mono float32 via soundfile."""
    raw = audio.get("bytes")
    source = io.BytesIO(raw) if raw is not None else audio["path"]
    data, sr = sf.read(source, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = soxr.resample(mono, sr, SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


def mswc_peek(*, dataset: str, config: str, split: str, n: int = 1) -> None:
    """Print the fields of the first ``n`` streamed examples (schema/decode check)."""
    stream = _load_stream(dataset, config, split)
    for i, example in enumerate(stream):
        if i >= n:
            break
        print(f"keys = {list(example.keys())}")
        for key in ("keyword", "word", "label"):
            if key in example:
                print(f"  candidate word field {key!r} = {example[key]!r}")
        audio = example.get("audio")
        if isinstance(audio, dict):
            print(f"  audio fields = {list(audio.keys())}, bytes={'yes' if audio.get('bytes') else 'no'}")
            try:
                wav = _decode_16k_mono(audio)
                print(f"  decoded OK -> {len(wav)} samples @ 16 kHz mono")
            except Exception as exc:
                print(f"  decode FAILED: {exc!r}")


def prepare_mswc(
    root: str | Path,
    *,
    n_words: int,
    clips_per_word: int,
    dataset: str = "MLCommons/ml_spoken_words",
    config: str = "en_opus",
    split: str = "train",
    word_key: str = "keyword",
    vocab_total: int = 20_000,
    min_clips: int = 10,
    max_stream: int = 0,
    seed: int = 0,
) -> list[str]:
    """Stream MSWC and materialize a hash-selected, alphabet-spread word subset.

    Words are chosen by a hash (a uniform ~``n_words / vocab_total`` fraction) so the
    kept vocabulary spans the whole alphabet, not just early letters. Each kept word
    is filled to ``clips_per_word``; words with fewer than ``min_clips`` are pruned.
    This streams the whole ``split`` (set ``max_stream`` to cap it, trading alphabet
    coverage for speed). Audio is decoded with soundfile + soxr. Delete ``root`` to
    restart.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = _load_stream(dataset, config, split)
    selector = _WordSelector(n_words, vocab_total, seed)

    counts: dict[str, int] = {}
    for i, example in enumerate(stream):
        if max_stream and i >= max_stream:
            break
        word = example[word_key]
        if not selector.keep(word):
            continue
        count = counts.get(word, 0)
        if count >= clips_per_word:
            continue
        out = root / word / f"{count:04d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, _decode_16k_mono(example["audio"]), SAMPLE_RATE)
        counts[word] = count + 1

    kept = sorted(word for word, count in counts.items() if count >= min_clips)
    if not kept:
        raise RuntimeError(f"no words reached {min_clips} clips from {dataset}/{config}; check --config/--word-key")
    pruned = 0
    for word, count in counts.items():
        if count < min_clips:
            shutil.rmtree(root / word, ignore_errors=True)
            pruned += 1
    total = sum(counts[word] for word in kept)
    print(f"materialized {total} clips across {len(kept)} words under {root} (pruned {pruned} under {min_clips} clips)")
    return kept


def mswc_samplers(
    root: str | Path,
    *,
    n_held_out: int = 50,
    limit: int | None = None,
    seed: int = 0,
    augment: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Return (train_sampler, held_out_sampler) over a prepared MSWC directory.

    Reserves ``n_held_out`` random words (seeded) for the few-shot probe.
    ``augment`` (if given) perturbs training clips only; see ``split_samplers``.
    """
    index = index_clips(root)
    if len(index) <= n_held_out:
        raise RuntimeError(f"only {len(index)} words under {root}; need more than n_held_out={n_held_out}")
    rng = np.random.default_rng(seed)
    words = sorted(index)
    held_out = [words[int(i)] for i in rng.choice(len(words), size=n_held_out, replace=False)]
    return split_samplers(index, held_out_words=held_out, limit=limit, seed=seed, augment=augment)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare an MSWC subset for backbone training (Phase 2).")
    p.add_argument("--root", default="data/mswc", help="output dir for materialized clips")
    p.add_argument("--n-words", type=int, default=500)
    p.add_argument("--clips-per-word", type=int, default=80)
    p.add_argument("--dataset", default="MLCommons/ml_spoken_words")
    p.add_argument("--config", default="en_opus", help="MSWC config, e.g. en_opus (smaller) or en_wav")
    p.add_argument("--split", default="train")
    p.add_argument("--word-key", default="keyword")
    p.add_argument("--vocab-total", type=int, default=20_000, help="estimated corpus vocab size (targets --n-words)")
    p.add_argument("--min-clips", type=int, default=10, help="prune words with fewer clips than this")
    p.add_argument("--max-stream", type=int, default=0, help="cap clips streamed (0 = whole split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inspect", action="store_true", help="print the schema of one streamed example and exit")
    args = p.parse_args()

    if args.inspect:
        mswc_peek(dataset=args.dataset, config=args.config, split=args.split)
        return
    prepare_mswc(
        args.root,
        n_words=args.n_words,
        clips_per_word=args.clips_per_word,
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        word_key=args.word_key,
        vocab_total=args.vocab_total,
        min_clips=args.min_clips,
        max_stream=args.max_stream,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
