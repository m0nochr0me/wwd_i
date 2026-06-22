"""Random-sampled audio pools from Hugging Face datasets (Phase 2/3 augmentation).

Background-noise augmentation (backbone ``--aug-bg-dir``) and head background
negatives need a pool of generic audio, but the source corpora are huge (e.g.
``benjamin-paine/freesound-laion-640k`` is 640k clips). Rather than download a whole
corpus, this **streams** it, shuffles with a seed, and materializes only ``n_clips``
decoded 16 kHz mono wavs — bounded bandwidth, fully reproducible. The kept wavs are
written as ``{out}/{i:05d}.wav`` and consumed by ``backgrounds.py`` / ``preprocess_bg``
like any other audio dir.

`datasets` is a training-only dep (imported lazily). Run
``python -m wwd_i.data.hf_audio --inspect --dataset <id>`` to confirm the audio column
and decoding before a large sample. See docs/architecture.md §7.
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.mswc import _decode_16k_mono  # shared HF-audio decode (soundfile + soxr)


def _stream(dataset: str, config: str | None, split: str):
    """Streaming HF dataset with its declared feature schema dropped (raw columns).

    We decode the audio bytes ourselves (soundfile + soxr), so the declared schema is
    unnecessary — and on WebDataset corpora it is actively harmful: ``0x3/vocal-bursts``
    stores FLAC bytes under a ``flac`` column while its metadata declares an ``audio``
    column, so ``datasets`` raises ``CastError`` mid-stream casting each shard to the
    mismatched features. Setting ``info.features = None`` skips that cast; iteration then
    yields the physical columns ({bytes|path} dict for parquet Audio, raw bytes for
    WebDataset members). Dropping the schema also avoids datasets' librosa-based Audio
    decoder (no py3.14 wheel). Pick the audio column with ``--audio-key`` (``--inspect``
    lists the columns).
    """
    import datasets

    from wwd_i.data._datasets_compat import patch_datasets_py314

    patch_datasets_py314()  # py3.14: fix datasets' _batch_setitems fingerprint override
    ds = datasets.load_dataset(dataset, config, split=split, streaming=True)
    ds.info.features = None
    return ds


# Audio column names vary by source: parquet Audio is named "audio"; a WebDataset names each
# member by its file extension, so the audio is under "flac"/"wav"/... (e.g. 0x3/vocal-bursts
# uses "flac"). --audio-key is a hint; the real column is auto-detected when it isn't present.
_AUDIO_KEY_HINTS = ("audio", "flac", "wav", "mp3", "ogg", "opus", "m4a", "aac", "wave")


def _is_audio_value(value) -> bool:
    """True if ``value`` holds decodable audio: raw bytes or a {bytes|path} dict."""
    return isinstance(value, (bytes, bytearray)) or (isinstance(value, dict) and ("bytes" in value or "path" in value))


def _pick_audio_key(example: dict, requested: str) -> str:
    """Resolve the audio column, tolerating WebDataset extension-named columns.

    Prefers ``requested``; else the first known audio-extension column; else any other
    column holding audio bytes. Raises a key-listing ``KeyError`` when none qualifies.
    """
    if _is_audio_value(example.get(requested)):
        return requested
    for key in _AUDIO_KEY_HINTS:
        if _is_audio_value(example.get(key)):
            return key
    for key, value in example.items():
        if key not in ("__key__", "__url__", "json") and _is_audio_value(value):
            return key
    raise KeyError(
        f"no audio column in {list(example.keys())} (requested --audio-key {requested!r}); "
        "pass --audio-key or run --inspect"
    )


def _write_pool(
    examples: Iterable[dict],
    out: str | Path,
    *,
    n_clips: int,
    audio_key: str,
    min_seconds: float,
    max_stream: int,
) -> tuple[int, int]:
    """Decode/filter/write up to ``n_clips`` clips from ``examples``. Returns (kept, skipped).

    Corrupt, too-short, or non-finite clips are skipped. ``max_stream`` caps examples
    consulted (0 = until ``n_clips`` reached).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    min_len = int(min_seconds * SAMPLE_RATE)
    kept = skipped = 0
    key = None
    for i, example in enumerate(examples):
        if kept >= n_clips or (max_stream and i >= max_stream):
            break
        if key is None:  # resolve the audio column once (the stream is homogeneous)
            key = _pick_audio_key(example, audio_key)
            if key != audio_key:
                print(f"audio column {audio_key!r} not present; using {key!r}")
        raw = example[key]
        try:
            # WebDataset members arrive as raw bytes; parquet Audio as a {bytes|path} dict.
            wav = _decode_16k_mono(raw if isinstance(raw, dict) else {"bytes": raw})
        except Exception:
            skipped += 1  # corrupt / unsupported codec
            continue
        if len(wav) < min_len or not np.all(np.isfinite(wav)):
            skipped += 1
            continue
        sf.write(out / f"{kept:05d}.wav", wav, SAMPLE_RATE)
        kept += 1
    return kept, skipped


def prepare_audio_pool(
    out: str | Path,
    *,
    dataset: str,
    n_clips: int,
    config: str | None = None,
    split: str = "train",
    audio_key: str = "audio",
    seed: int = 0,
    buffer_size: int = 2000,
    min_seconds: float = 1.0,
    max_stream: int = 0,
) -> int:
    """Stream ``dataset``, shuffle (seeded), and materialize ``n_clips`` 16 kHz wavs.

    Only ~``n_clips`` examples (plus the shuffle buffer) are pulled from the hub, so a
    640k-clip corpus costs the bandwidth of the subset, not the whole thing. The shuffle
    is a windowed (``buffer_size``) shuffle over the stream, not a uniform draw over the
    full corpus — ``seed`` makes it reproducible. Returns the count written.
    """
    stream = _stream(dataset, config, split).shuffle(seed=seed, buffer_size=buffer_size)
    kept, skipped = _write_pool(
        stream, out, n_clips=n_clips, audio_key=audio_key, min_seconds=min_seconds, max_stream=max_stream
    )
    print(f"materialized {kept} clips under {out} (skipped {skipped}; seed {seed}, buffer {buffer_size})")
    if kept < n_clips:
        print(f"  note: only {kept}/{n_clips} kept — raise --buffer-size/--max-stream or lower --min-seconds")
    return kept


def inspect_pool(
    dataset: str, *, config: str | None = None, split: str = "train", audio_key: str = "audio", n: int = 1
) -> None:
    """Print the fields of the first ``n`` streamed examples and decode one clip."""
    stream = _stream(dataset, config, split)
    for i, example in enumerate(stream):
        if i >= n:
            break
        print(f"keys = {list(example.keys())}")
        try:
            key = _pick_audio_key(example, audio_key)
        except KeyError as exc:
            print(f"  {exc}")
            continue
        if key != audio_key:
            print(f"  auto-detected audio column {key!r} (--audio-key {audio_key!r} not present)")
        raw = example[key]
        audio = raw if isinstance(raw, dict) else {"bytes": raw}  # WebDataset member -> wrap bytes
        print(f"  audio fields = {list(audio.keys())}, bytes={'yes' if audio.get('bytes') else 'no'}")
        try:
            wav = _decode_16k_mono(audio)
            print(f"  decoded OK -> {len(wav)} samples @ 16 kHz mono ({len(wav) / SAMPLE_RATE:.2f}s)")
        except Exception as exc:
            print(f"  decode FAILED: {exc!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Random-sample an audio pool from a Hugging Face dataset (Phase 2/3).")
    p.add_argument("--dataset", required=True, help="HF dataset id, e.g. benjamin-paine/freesound-laion-640k")
    p.add_argument("--out", default="data/audio_pool", help="output dir for the sampled 16 kHz wavs")
    p.add_argument("--n-clips", type=int, default=2000, help="number of clips to materialize")
    p.add_argument("--config", default=None, help="dataset config/subset (default: the dataset default)")
    p.add_argument("--split", default="train")
    p.add_argument("--audio-key", default="audio", help="column holding the audio (see --inspect)")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed (reproducible subset)")
    p.add_argument("--buffer-size", type=int, default=2000, help="streaming shuffle-buffer size")
    p.add_argument("--min-seconds", type=float, default=1.0, help="skip clips shorter than this")
    p.add_argument("--max-stream", type=int, default=0, help="cap examples consulted (0 = until n-clips reached)")
    p.add_argument("--inspect", action="store_true", help="print one example's schema + decode check, then exit")
    args = p.parse_args()

    if args.inspect:
        inspect_pool(args.dataset, config=args.config, split=args.split, audio_key=args.audio_key)
        return
    prepare_audio_pool(
        args.out,
        dataset=args.dataset,
        n_clips=args.n_clips,
        config=args.config,
        split=args.split,
        audio_key=args.audio_key,
        seed=args.seed,
        buffer_size=args.buffer_size,
        min_seconds=args.min_seconds,
        max_stream=args.max_stream,
    )


if __name__ == "__main__":
    main()
