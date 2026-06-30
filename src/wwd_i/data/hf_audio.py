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
from typing import cast

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
    ds = cast(datasets.IterableDataset, datasets.load_dataset(dataset, config, split=split, streaming=True))
    ds.info.features = None
    return ds


# Audio column names vary by source: parquet Audio is named "audio"; a WebDataset names each
# member by file extension ("flac"/"wav"/...). Some WebDatasets store audio and metadata
# UNPAIRED — 0x3/vocal-bursts ("one .tar.gz per category") streams FLAC audio and the JSON
# metadata as SEPARATE examples, so an example may hold audio, or only JSON, or neither.
# --audio-key is a hint; the real column is auto-detected and audio-less examples are skipped.
_AUDIO_KEY_HINTS = ("audio", "flac", "wav", "mp3", "ogg", "opus", "m4a", "aac", "wave")


def _is_audio_value(value) -> bool:
    """True if ``value`` holds decodable audio: raw bytes or a {bytes|path} dict."""
    return isinstance(value, (bytes, bytearray)) or (isinstance(value, dict) and ("bytes" in value or "path" in value))


def _find_audio(example: dict, requested: str) -> tuple[str, object] | None:
    """Return ``(column, value)`` of the audio in ``example``, or ``None`` if it has none.

    Prefers ``requested``, then a known audio-extension column, then any other column holding
    audio bytes. ``None`` (not an error) lets callers skip metadata-only examples.
    """
    for key in (requested, *_AUDIO_KEY_HINTS):
        if _is_audio_value(example.get(key)):
            return key, example[key]
    for key, value in example.items():
        if key not in ("__key__", "__url__", "json") and _is_audio_value(value):
            return key, value
    return None


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
    for i, example in enumerate(examples):
        if kept >= n_clips or (max_stream and i >= max_stream):
            break
        found = _find_audio(example, audio_key)
        if found is None:
            skipped += 1  # metadata-only example (unpaired WebDataset entry) — no audio to decode
            continue
        raw = found[1]
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


def _audio_column(schema, requested: str) -> str:
    """Pick the audio column of a parquet ``schema``: an Audio struct ({bytes|path}) or audio bytes.

    Prefers ``requested``, then a known audio-extension name, then any struct-with-bytes/path column.
    """
    import pyarrow as pa

    def kind(name: str) -> str | None:
        field_type = schema.field(name).type
        if pa.types.is_struct(field_type) and any(f.name in ("bytes", "path") for f in field_type):
            return "struct"
        if pa.types.is_binary(field_type) or pa.types.is_large_binary(field_type):
            return "binary"
        return None

    names = set(schema.names)
    for key in (requested, *_AUDIO_KEY_HINTS):
        if key in names and kind(key):
            return key
    for name in schema.names:  # fall back to any embedded-audio struct column
        if kind(name) == "struct":
            return name
    raise RuntimeError(f"no audio column (struct with bytes/path, or audio-named binary) in {schema.names}")


def _download_parquet_shards(dataset: str, config: str | None, split: str) -> list[str]:
    """Download a hub dataset's parquet shard(s) for one ``split`` to the local HF cache.

    Reads the hub's auto-converted parquet (``refs/convert/parquet``), laid out as
    ``{config}/{split}/{shard}.parquet``. Returns local file paths.
    """
    from huggingface_hub import HfApi, hf_hub_download

    revision = "refs/convert/parquet"
    parquet = [
        f for f in HfApi().list_repo_files(dataset, repo_type="dataset", revision=revision) if f.endswith(".parquet")
    ]
    if config is None:
        configs = sorted({f.split("/")[0] for f in parquet})
        if len(configs) != 1:
            raise RuntimeError(f"{dataset} parquet has multiple configs {configs}; pass --config")
        config = configs[0]
    prefix = f"{config}/{split}/"
    shards = sorted(f for f in parquet if f.startswith(prefix))
    if not shards:
        raise RuntimeError(f"no parquet shards under {prefix!r} in {dataset}@{revision}; check --config/--split")
    return [hf_hub_download(dataset, filename=f, repo_type="dataset", revision=revision) for f in shards]


def _write_pool_parquet(
    paths: Iterable[str | Path],
    out: str | Path,
    *,
    n_clips: int,
    audio_key: str,
    min_seconds: float,
    max_stream: int,
    seed: int,
) -> tuple[int, int]:
    """Low-RAM local read: iterate parquet row groups (one resident at a time). Returns (kept, skipped).

    HF *streaming* materializes a whole row group per step (``parquet.py`` ``batch_size`` defaults to
    the row-group row count), so a single-shard, huge-row-group Audio corpus — e.g. ``google/fleurs``
    (2602 rows in one 1.7 GB shard of 3x ~650 MB row groups) — OOMs Colab regardless of ``--buffer-size``.
    Reading the downloaded shard row-group-by-row-group with only the audio column projected bounds peak
    RAM to ~one row group. Row-group and within-group order are seeded-shuffled for a reproducible subset.
    Skips short/corrupt/non-finite clips, same as ``_write_pool``.
    """
    import gc

    import pyarrow.parquet as pq

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    min_len = int(min_seconds * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    kept = skipped = consulted = 0
    for path in paths:
        if kept >= n_clips or (max_stream and consulted >= max_stream):
            break
        pf = pq.ParquetFile(path)
        col_name = _audio_column(pf.schema_arrow, audio_key)
        for rg in rng.permutation(pf.num_row_groups):
            if kept >= n_clips or (max_stream and consulted >= max_stream):
                break
            table = pf.read_row_group(int(rg), columns=[col_name])  # ~one row group's audio column in RAM
            column = table.column(col_name)
            for j in rng.permutation(table.num_rows):
                if kept >= n_clips or (max_stream and consulted >= max_stream):
                    break
                consulted += 1
                value = column[int(j)].as_py()  # {bytes|path} dict (Audio struct) or raw bytes (binary col)
                try:
                    wav = _decode_16k_mono(value if isinstance(value, dict) else {"bytes": value})
                except Exception:
                    skipped += 1  # corrupt / unsupported codec
                    continue
                if len(wav) < min_len or not np.all(np.isfinite(wav)):
                    skipped += 1
                    continue
                sf.write(out / f"{kept:05d}.wav", wav, SAMPLE_RATE)
                kept += 1
            del table, column
            gc.collect()  # return the row group's buffers to the OS before reading the next
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
    download: bool = False,
) -> int:
    """Stream ``dataset``, shuffle (seeded), and materialize ``n_clips`` 16 kHz wavs.

    Only ~``n_clips`` examples (plus the shuffle buffer) are pulled from the hub, so a
    640k-clip corpus costs the bandwidth of the subset, not the whole thing. The shuffle
    is a windowed (``buffer_size``) shuffle over the stream, not a uniform draw over the
    full corpus — ``seed`` makes it reproducible. Returns the count written.

    ``download=True`` fetches the parquet shard(s) to disk and reads them row-group-by-row-group
    instead of streaming — the low-RAM path for single-shard, huge-row-group Audio parquet (e.g.
    ``google/fleurs``), whose streaming OOMs because it materializes a whole ~650 MB row group per
    step. No bandwidth penalty when the subset is most of a single shard.
    """
    if download:
        paths = _download_parquet_shards(dataset, config, split)
        kept, skipped = _write_pool_parquet(
            paths, out, n_clips=n_clips, audio_key=audio_key, min_seconds=min_seconds, max_stream=max_stream, seed=seed
        )
    else:
        stream = _stream(dataset, config, split).shuffle(seed=seed, buffer_size=buffer_size)
        kept, skipped = _write_pool(
            stream, out, n_clips=n_clips, audio_key=audio_key, min_seconds=min_seconds, max_stream=max_stream
        )
    mode = "downloaded" if download else f"streamed, buffer {buffer_size}"
    print(f"materialized {kept} clips under {out} (skipped {skipped}; seed {seed}, {mode})")
    if kept < n_clips:
        print(f"  note: only {kept}/{n_clips} kept — raise --buffer-size/--max-stream or lower --min-seconds")
    return kept


def inspect_pool(
    dataset: str, *, config: str | None = None, split: str = "train", audio_key: str = "audio", n: int = 1
) -> None:
    """Print the keys of the first ``n`` streamed examples and decode the first audio clip found.

    Some streams (e.g. 0x3/vocal-bursts) interleave audio with metadata-only examples, so the
    first example may carry no audio; a bounded prefix is scanned to find and decode-check one.
    """
    stream = _stream(dataset, config, split)
    scan = max(n, 500)
    for i, example in enumerate(stream):
        if i < n:
            print(f"keys = {list(example.keys())}")
        found = _find_audio(example, audio_key)
        if found is not None:
            key, raw = found
            audio = raw if isinstance(raw, dict) else {"bytes": raw}  # WebDataset member -> wrap bytes
            note = "" if key == audio_key else f" (auto-detected; --audio-key {audio_key!r} absent)"
            print(f"audio column = {key!r}{note}; bytes={'yes' if audio.get('bytes') else 'no'}")
            try:
                wav = _decode_16k_mono(audio)
                print(f"  decoded OK -> {len(wav)} samples @ 16 kHz mono ({len(wav) / SAMPLE_RATE:.2f}s)")
            except Exception as exc:
                print(f"  decode FAILED: {exc!r}")
            return
        if i + 1 >= scan:
            break
    print(f"no audio-bearing example in the first {scan} streamed — check --audio-key or the dataset layout")


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
    p.add_argument(
        "--download",
        action="store_true",
        help="fetch parquet shard(s) to disk and read row-group-by-row-group (low-RAM path for "
        "single-shard Audio parquet like google/fleurs, whose streaming OOMs)",
    )
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
        download=args.download,
    )


if __name__ == "__main__":
    main()
