"""MSWC (Multilingual Spoken Words) scale-up corpus (Phase 2).

Speech Commands proved the pipeline on ~24 training words; MSWC scales the
vocabulary into the hundreds/thousands so the frozen embedding generalizes to
arbitrary wake words. The corpus is streamed from the Hugging Face hub
(``MLCommons/ml_spoken_words``) so we pull only a capped subset rather than the
full multi-GB tarball, materialize it as 16 kHz ``{word}/{n}.wav`` clips under
``root``, then train from that directory with the shared ``clips.py`` path.

`datasets` is a training-only dep (the `train` group), imported lazily. Run
``uv run --group train python -m wwd_i.data.mswc --inspect`` to verify the dataset
schema before a large prepare. See docs/implementation-plan.md Phase 2.
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.clips import index_clips, split_samplers
from wwd_i.data.episodic import EpisodicSampler


class _Subset:
    """Caps a stream to ``n_words`` words and ``clips_per_word`` clips each."""

    def __init__(self, n_words: int, clips_per_word: int) -> None:
        self.n_words = n_words
        self.clips_per_word = clips_per_word
        self.counts: dict[str, int] = {}

    def take(self, word: str) -> int | None:
        """Return the next local index to write for ``word``, or None to skip it."""
        count = self.counts.get(word)
        if count is None:
            if len(self.counts) >= self.n_words:
                return None  # vocabulary full; ignore unseen words
            count = self.counts[word] = 0
        if count >= self.clips_per_word:
            return None
        self.counts[word] = count + 1
        return count

    @property
    def done(self) -> bool:
        return len(self.counts) >= self.n_words and all(c >= self.clips_per_word for c in self.counts.values())


def _load_stream(dataset: str, config: str, split: str):
    import datasets

    try:  # older script-based datasets need opt-in remote code; newer ones reject the kwarg
        return datasets.load_dataset(dataset, config, split=split, streaming=True, trust_remote_code=True)
    except TypeError:
        return datasets.load_dataset(dataset, config, split=split, streaming=True)


def mswc_peek(*, dataset: str, config: str, split: str, n: int = 1) -> None:
    """Print the fields of the first ``n`` streamed examples (schema sanity check)."""
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
            print(f"  audio.sampling_rate = {audio.get('sampling_rate')}, samples = {len(audio.get('array', []))}")


def prepare_mswc(
    root: str | Path,
    *,
    n_words: int,
    clips_per_word: int,
    dataset: str = "MLCommons/ml_spoken_words",
    config: str = "en",
    split: str = "train",
    word_key: str = "keyword",
    max_stream: int = 2_000_000,
) -> list[str]:
    """Stream a capped MSWC subset to ``root`` as 16 kHz ``{word}/{n}.wav`` clips.

    Returns the materialized word list. Resampling to 16 kHz is delegated to the
    `datasets` Audio decoder. Idempotent only at the file level (re-running
    overwrites clip files); delete ``root`` to start clean.
    """
    import datasets

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = _load_stream(dataset, config, split).cast_column("audio", datasets.Audio(sampling_rate=SAMPLE_RATE))

    subset = _Subset(n_words, clips_per_word)
    written = 0
    for i, example in enumerate(stream):
        if i >= max_stream:
            break
        index = subset.take(example[word_key])
        if index is None:
            if subset.done:
                break
            continue
        wav = np.asarray(example["audio"]["array"], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        out = root / example[word_key] / f"{index:04d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, wav, SAMPLE_RATE)
        written += 1
        if subset.done:
            break

    words = sorted(word for word, count in subset.counts.items() if count > 0)
    if not words:
        raise RuntimeError(f"no clips materialized from {dataset}/{config}; check --config/--word-key with --inspect")
    print(f"materialized {written} clips across {len(words)} words under {root}")
    return words


def mswc_samplers(
    root: str | Path, *, n_held_out: int = 50, limit: int | None = None, seed: int = 0
) -> tuple[EpisodicSampler, EpisodicSampler]:
    """Return (train_sampler, held_out_sampler) over a prepared MSWC directory.

    Reserves ``n_held_out`` random words (seeded) for the few-shot probe.
    """
    index = index_clips(root)
    if len(index) <= n_held_out:
        raise RuntimeError(f"only {len(index)} words under {root}; need more than n_held_out={n_held_out}")
    rng = np.random.default_rng(seed)
    words = sorted(index)
    held_out = [words[int(i)] for i in rng.choice(len(words), size=n_held_out, replace=False)]
    return split_samplers(index, held_out_words=held_out, limit=limit, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare an MSWC subset for backbone training (Phase 2).")
    p.add_argument("--root", default="data/mswc", help="output dir for materialized clips")
    p.add_argument("--n-words", type=int, default=500)
    p.add_argument("--clips-per-word", type=int, default=80)
    p.add_argument("--dataset", default="MLCommons/ml_spoken_words")
    p.add_argument("--config", default="en")
    p.add_argument("--split", default="train")
    p.add_argument("--word-key", default="keyword")
    p.add_argument("--max-stream", type=int, default=2_000_000)
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
        max_stream=args.max_stream,
    )


if __name__ == "__main__":
    main()
