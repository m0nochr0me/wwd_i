"""Google Gemini TTS backend (Phase 3) — opt-in.

Synthesizes diverse spoken renderings of a phrase via the Gemini speech-generation
API — the 30 prebuilt voices (the dominant timbral lever) crossed with a small set
of natural-language delivery directives (Gemini's only prosody knob; there are no
numeric stability/style settings like ElevenLabs). The generic machinery (caching,
normalization, file layout, per-voice skip) lives in ``tts.py``; this module is
just the Gemini ``TtsBackend``: enumerate the voice × style grid and call the SDK.

**Opt-in / licensing.** The positives train an ML model (the head), so the backend's
terms must permit that use. Google's Gemini API terms restrict using output to build
competing models and vary by tier (free vs paid data use) — verify your tier's
current terms before training a *distributed* head. ``google-genai`` is a
generation-only dep (the ``train`` group), never shipped. See ``docs/data-licensing.md``.

The API key is read from ``GEMINI_API_KEY``; nothing secret is stored in code. The
single SDK-coupled seam is one synthesis call (``_synthesize``), injectable for tests;
the voice/style grid is a fixed constant, so ``variants`` needs no network. Gemini
returns s16le mono PCM @ 24 kHz, which we resample to ``SAMPLE_RATE``. Run ``--smoke``
(one clip) to validate the SDK before a big batch.
"""

import argparse
import os

import numpy as np
import soxr

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.tts import TtsBackend, Variant, generate_clips

DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_SAMPLE_RATE = 24_000  # Gemini TTS returns s16le mono PCM @ 24 kHz; we resample to SAMPLE_RATE

# The 30 prebuilt Gemini voices — the dominant timbral diversity lever.
VOICES: tuple[str, ...] = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)  # fmt: skip

# Natural-language delivery directives prepended as "{style}: {phrase}" — Gemini's
# prosody lever (it has no numeric stability/style knobs). "" = neutral (phrase sent
# verbatim). Crossed with VOICES for prosodic spread; kept conservative so the
# directive shapes delivery without leaking spoken words into the clip.
STYLES: tuple[str, ...] = (
    "",
    "Say clearly",
    "Say cheerfully",
    "Say in a calm, natural tone",
    "Say slowly",
    "Say quickly",
    "Say in a serious tone",
)


def _client():  # pragma: no cover - thin SDK seam
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY in the environment")
    return genai.Client(api_key=key)


def _synthesize(client, text: str, voice: str, *, model: str) -> bytes:  # pragma: no cover - thin SDK seam
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice),
                ),
            ),
        ),
    )
    for part in resp.candidates[0].content.parts or []:
        blob = getattr(part, "inline_data", None)
        if blob is not None and blob.data:
            return blob.data  # SDK returns the PCM bytes already base64-decoded
    raise RuntimeError(f"Gemini returned no audio for voice {voice!r} (blocked or empty response)")


def _prompt(phrase: str, style: str) -> str:
    """Prepend the delivery directive Gemini interprets but doesn't speak; neutral = verbatim."""
    return f"{style}: {phrase}" if style else phrase


def _pcm_to_float16k(pcm: bytes) -> np.ndarray:
    """Decode s16le mono @ 24 kHz to float32 in [-1, 1], resampled to ``SAMPLE_RATE``."""
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if GEMINI_SAMPLE_RATE != SAMPLE_RATE:
        audio = soxr.resample(audio, GEMINI_SAMPLE_RATE, SAMPLE_RATE)
    return audio.astype(np.float32)


def _variants(n_clips: int, seed: int) -> list[tuple[str, str]]:
    """Spread ``n_clips`` over a shuffled voice × style grid (deterministic in ``seed``)."""
    grid = [(v, s) for v in VOICES for s in STYLES]
    np.random.default_rng(seed).shuffle(grid)
    return [grid[i % len(grid)] for i in range(n_clips)]


class GeminiTtsBackend(TtsBackend):
    """``TtsBackend`` over the Gemini speech-generation API. ``client`` / ``synthesize``
    are injectable seams (default to the real SDK) so the offline logic stays testable.

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
        pcm = self._synthesize(self._client(), text, variant.voice, model=self.model)
        return _pcm_to_float16k(pcm)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Gemini TTS samples for a phrase (Phase 3; opt-in).")
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

    backend = GeminiTtsBackend(model=args.model)
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
