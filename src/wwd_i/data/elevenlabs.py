"""ElevenLabs v3 TTS sample generation (Phase 3).

Synthesizes diverse spoken renderings of a phrase — many voices crossed with a
few prosody settings — and caches them as 16 kHz mono wavs, RMS-normalized.
Used for both positives (the wake phrase) and hard negatives (near phrases).

The API key is read from ``ELEVENLABS_API_KEY``; nothing secret is stored in
code. ``elevenlabs`` is a generation-only dep (the ``train`` group), imported
lazily. The two network seams — listing voices and one synthesis call — are the
only SDK-coupled code (``_client`` / ``_synthesize``); everything else (caching,
PCM decode, normalization, voice/settings spread, file layout) is offline and
unit-tested. Run ``--smoke`` (one clip) to validate the SDK before a big batch.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from wwd_i.config import SAMPLE_RATE

DEFAULT_MODEL = "eleven_v3"
PCM_FORMAT = "pcm_16000"  # raw s16le mono @ 16 kHz — no resample needed
STABILITIES = (0.0, 0.5, 1.0)  # eleven_v3 only accepts discrete stability: Creative / Natural / Robust
STYLES = (0.0, 0.3, 0.6)


def _client():  # pragma: no cover - thin SDK seam
    from elevenlabs.client import ElevenLabs

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("set ELEVENLABS_API_KEY in the environment")
    return ElevenLabs(api_key=key)


def _list_voice_ids(client, max_voices: int) -> list[str]:
    return [v.voice_id for v in client.voices.get_all().voices][:max_voices]


def _synthesize(
    client, text: str, voice_id: str, *, model: str, stability: float, style: float
) -> bytes:  # pragma: no cover - thin SDK seam
    from elevenlabs import VoiceSettings

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
        output_format=PCM_FORMAT,
        voice_settings=VoiceSettings(stability=stability, similarity_boost=0.75, style=style),
    )
    return b"".join(audio)  # the SDK streams byte chunks


def _pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _rms_normalize(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    if rms < 1e-6:
        return x.astype(np.float32)  # silence — leave as-is
    y = x * (10.0 ** (target_dbfs / 20.0) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def _variants(n_clips: int, voice_ids: list[str], seed: int) -> list[tuple[str, float, float]]:
    """Spread ``n_clips`` over a shuffled voice × stability × style grid."""
    grid = [(v, st, sy) for v in voice_ids for st in STABILITIES for sy in STYLES]
    np.random.default_rng(seed).shuffle(grid)
    return [grid[i % len(grid)] for i in range(n_clips)]


def generate_clips(
    phrase: str,
    out_dir: str | Path,
    *,
    n_clips: int = 300,
    model: str = DEFAULT_MODEL,
    max_voices: int = 40,
    seed: int = 0,
    client=None,
    synthesize=_synthesize,
) -> list[Path]:
    """Synthesize ``n_clips`` diverse renderings of ``phrase`` into ``out_dir``.

    Cached by (phrase, voice, settings): existing files are reused, so reruns and
    a bumped ``n_clips`` only synthesize the new variants. ``client`` /
    ``synthesize`` are injectable seams (default to the real SDK).
    """
    client = client or _client()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_ids = _list_voice_ids(client, max_voices)
    if not voice_ids:
        raise RuntimeError("no voices available on this ElevenLabs account")

    written: list[Path] = []
    for voice_id, stability, style in _variants(n_clips, voice_ids, seed):
        tag = hashlib.blake2b(f"{phrase}|{voice_id}|{stability}|{style}".encode(), digest_size=8).hexdigest()
        path = out_dir / f"{tag}.wav"
        if not path.exists():
            pcm = synthesize(client, phrase, voice_id, model=model, stability=stability, style=style)
            sf.write(path, _rms_normalize(_pcm16_to_float(pcm)), SAMPLE_RATE)
        if path not in written:
            written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ElevenLabs v3 samples for a phrase (Phase 3).")
    p.add_argument("--phrase", help="text to synthesize, e.g. 'hey computer'")
    p.add_argument("--hard-negs-for", help="instead synthesize the hard-negative near-phrases for this wake phrase")
    p.add_argument("--out", required=True, help="output dir for the cached wavs")
    p.add_argument("--n-clips", type=int, default=300)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-voices", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="synthesize a single clip to validate the SDK, then exit")
    args = p.parse_args()
    if not (args.phrase or args.hard_negs_for):
        p.error("pass --phrase or --hard-negs-for")

    common = {"model": args.model, "max_voices": args.max_voices, "seed": args.seed}
    if args.hard_negs_for:
        from wwd_i.data.negatives import hard_negative_phrases

        phrases = hard_negative_phrases(args.hard_negs_for)
        written = [c for ph in phrases for c in generate_clips(ph, args.out, n_clips=args.n_clips, **common)]
        print(f"wrote {len(written)} hard-negative clips across {len(phrases)} phrases under {args.out}")
        return

    n_clips = 1 if args.smoke else args.n_clips
    paths = generate_clips(args.phrase, args.out, n_clips=n_clips, **common)
    print(f"wrote {len(paths)} clip(s) for {args.phrase!r} under {args.out}")


if __name__ == "__main__":
    main()
