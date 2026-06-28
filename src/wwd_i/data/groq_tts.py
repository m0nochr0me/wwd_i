"""Groq Orpheus TTS backend (Phase 3) — opt-in.

Synthesizes diverse spoken renderings of a phrase via Groq's hosted Orpheus speech
API — the 6 English prebuilt voices (the dominant timbral lever) crossed with a small
set of bracketed vocal directions (Orpheus's English-only prosody knob: ``[cheerful]``,
``[calm]``, ... shape delivery without being spoken). The generic machinery (caching,
normalization, file layout, per-voice skip) lives in ``tts.py``; this module is just
the Groq ``TtsBackend``: enumerate the voice × style grid and call the SDK.

**Opt-in / licensing.** The positives train an ML model (the head), so the backend's
terms must permit that use. Orpheus's open weights are Apache-2.0 (which permits
training on output), but access here is via Groq's hosted API — confirm Groq's terms
of service cover training before distributing a head. ``groq`` is a generation-only
dep (the ``train`` group), never shipped. See ``docs/data-licensing.md``.

The API key is read from ``GROQ_API_KEY``; nothing secret is stored in code. The single
SDK-coupled seam is one synthesis call (``_synthesize``), injectable for tests; the
voice/style grid is a fixed constant, so ``variants`` needs no network. Orpheus returns
a WAV whose header carries its own sample rate, which we decode and resample to
``SAMPLE_RATE``. Run ``--smoke`` (one clip) to validate the SDK before a big batch.
"""

import argparse
import io
import os

import numpy as np
import soundfile as sf
import soxr

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.tts import TtsBackend, Variant, generate_clips

DEFAULT_MODEL = "canopylabs/orpheus-v1-english"

# The 6 English Orpheus voices — the dominant timbral diversity lever. Lowercase ids
# match the API (the docs' example passes voice="troy").
VOICES: tuple[str, ...] = ("autumn", "diana", "hannah", "austin", "daniel", "troy")

# Bracketed vocal directions (Orpheus's English-only prosody lever) prepended as
# "{style} {phrase}". "" = neutral (phrase sent verbatim). The model reads these as
# delivery cues without speaking them; kept to tone descriptors (never sound-effect
# tags like <laugh>, which inject audible non-speech) so nothing leaks into the clip.
STYLES: tuple[str, ...] = ("", "[calm]", "[cheerful]", "[serious]", "[excited]", "[friendly]")


def _client():  # pragma: no cover - thin SDK seam
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("set GROQ_API_KEY in the environment")
    return Groq(api_key=key)


def _synthesize(client, text: str, voice: str, *, model: str) -> bytes:  # pragma: no cover - thin SDK seam
    resp = client.audio.speech.create(model=model, voice=voice, input=text, response_format="wav")
    return resp.read()  # OpenAI-compatible binary response -> raw WAV bytes


def _prompt(phrase: str, style: str) -> str:
    """Prepend the bracketed direction Orpheus interprets but doesn't speak; neutral = verbatim."""
    return f"{style} {phrase}" if style else phrase


def _wav_to_float16k(wav: bytes) -> np.ndarray:
    """Decode WAV bytes to float32 mono in [-1, 1], resampled to ``SAMPLE_RATE``.

    The WAV header carries its own sample rate, so we resample off that rather than
    assume Orpheus's output rate.
    """
    audio, rate = sf.read(io.BytesIO(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE)
    return audio.astype(np.float32)


def _variants(n_clips: int, seed: int) -> list[tuple[str, str]]:
    """Spread ``n_clips`` over a shuffled voice × style grid (deterministic in ``seed``)."""
    grid = [(v, s) for v in VOICES for s in STYLES]
    np.random.default_rng(seed).shuffle(grid)
    return [grid[i % len(grid)] for i in range(n_clips)]


class GroqTtsBackend(TtsBackend):
    """``TtsBackend`` over Groq's Orpheus speech API. ``client`` / ``synthesize`` are
    injectable seams (default to the real SDK) so the offline logic stays testable.

    Unique clips cap at ``len(VOICES) * len(STYLES)`` (the grid); ``generate_clips``
    caches by (phrase, voice, style), so a larger ``n_clips`` just reuses cells.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, client=None, synthesize=_synthesize):
        self.model = model
        self._given_client = client
        self._synthesize = synthesize

    def _client(self):
        if self._given_client is None:
            self._given_client = _client()
        return self._given_client

    def variants(self, n_clips: int, *, seed: int) -> list[Variant]:
        return [Variant(tag=f"{v}|{s}", voice=v, params={"style": s}) for v, s in _variants(n_clips, seed)]

    def synthesize(self, phrase: str, variant: Variant) -> np.ndarray:
        text = _prompt(phrase, variant.params["style"])
        wav = self._synthesize(self._client(), text, variant.voice, model=self.model)
        return _wav_to_float16k(wav)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Groq Orpheus TTS samples for a phrase (Phase 3; opt-in).")
    p.add_argument("--phrase", help="text to synthesize, e.g. 'hey computer'")
    p.add_argument("--hard-negs-for", help="instead synthesize the hard-negative near-phrases for this wake phrase")
    p.add_argument(
        "--llm-confusables",
        type=int,
        default=0,
        metavar="N",
        help="with --hard-negs-for: also fetch N acoustic confusables from Gemini (needs GEMINI_API_KEY)",
    )
    p.add_argument("--out", required=True, help="output dir for the cached wavs")
    p.add_argument("--n-clips", type=int, default=300)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="synthesize a single clip to validate the SDK, then exit")
    args = p.parse_args()
    if not (args.phrase or args.hard_negs_for):
        p.error("pass --phrase or --hard-negs-for")

    backend = GroqTtsBackend(model=args.model)
    common = {"n_clips": args.n_clips, "seed": args.seed}
    if args.hard_negs_for:
        from wwd_i.data.negatives import hard_negative_phrases

        extra: list[str] = []
        if args.llm_confusables:
            from wwd_i.data.confusables import generate_confusables

            extra = generate_confusables(args.hard_negs_for, n=args.llm_confusables)
            print(f"+{len(extra)} LLM confusables: {', '.join(extra)}")
        phrases = hard_negative_phrases(args.hard_negs_for, extra)
        written = [c for ph in phrases for c in generate_clips(ph, args.out, backend, **common)]
        print(f"wrote {len(written)} hard-negative clips across {len(phrases)} phrases under {args.out}")
        return

    n_clips = 1 if args.smoke else args.n_clips
    paths = generate_clips(args.phrase, args.out, backend, n_clips=n_clips, seed=args.seed)
    print(f"wrote {len(paths)} clip(s) for {args.phrase!r} under {args.out}")


if __name__ == "__main__":
    main()
