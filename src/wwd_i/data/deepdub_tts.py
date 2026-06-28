"""Deepdub TTS backend (Phase 3) — opt-in.

Deepdub is a voice-cloning / dubbing API, not a fixed multi-voice catalog like the
other backends — ``list_voices()`` returns only an account's own prompts (often empty)
and there's no public preset catalog to enumerate. So diversity here comes from a
*configurable* list of ``voice_prompt_id`` strings (Deepdub's public presets by
default; add your console / cloned ids for real timbral spread) crossed with Deepdub's
prosody knobs — ``temperature`` and ``variance`` — plus a distinct per-clip ``seed`` so
a handful of voices still yields ``n_clips`` genuinely-distinct renderings. The generic
machinery (caching, normalization, file layout, per-voice skip) lives in ``tts.py``;
this module is just the Deepdub ``TtsBackend``: spread the grid and call the SDK.

**Opt-in / licensing.** The positives train an ML model (the head), so the backend's
terms must permit that use. Deepdub is a commercial dubbing service — treat it like
ElevenLabs (opt-in) and confirm its terms permit training on the output, especially
cloned-voice output, before distributing a head. ``deepdub`` is a generation-only dep
(the ``train`` group), never shipped. See ``docs/data-licensing.md``.

The API key is read from ``DEEPDUB_API_KEY``; nothing secret is stored in code. The
single SDK-coupled seam is one synthesis call (``_synthesize``), injectable for tests;
the voice/prosody grid is offline, so ``variants`` needs no network. The sync REST
``/tts`` endpoint serves **mp3** (``headerless-wav``/``opus`` are rejected there) and
honors ``sample_rate``, so we request mp3 @ ``SAMPLE_RATE`` and decode with soundfile
(mp3 decoding needs libsndfile ≥ 1.1 w/ mpg123, bundled by ``soundfile`` ≥ 0.14 — no
ffmpeg). Run ``--smoke`` (one clip) to validate the SDK before a big batch.
"""

import argparse
import io
import os

import numpy as np
import soundfile as sf
import soxr

from wwd_i.config import SAMPLE_RATE
from wwd_i.data.tts import TtsBackend, Variant, generate_clips

DEFAULT_MODEL = "dd-etts-3.2"
LOCALE = "en-US"
REQUEST_FORMAT = "mp3"  # the only decodable format the sync REST /tts serves (headerless-wav/opus rejected there)
REQUEST_SAMPLE_RATE = SAMPLE_RATE  # mp3 honors it -> decode lands at 16 kHz, no resample in practice

# Deepdub exposes no preset catalog via list_voices (it's cloning-oriented); these two
# public preset prompt ids are working defaults. Add your own (console / cloned) ids via
# --voice-prompt-id for real timbral diversity — one voice gives only prosodic spread.
DEFAULT_VOICE_PROMPT_IDS: tuple[str, ...] = (
    "5d3dc622-69bd-4c00-9513-05df47dbdea6_authoritative",
    "bd1b00bb-be1c-4679-8eaa-0fcbfd4ff773",
)

TEMPERATURES = (0.3, 0.6, 0.9)  # Deepdub output diversity (0-1); higher = more varied
VARIANCES = (0.3, 0.7)  # Deepdub voice-variation intensity (0-1)


def _client():  # pragma: no cover - thin SDK seam
    from deepdub import DeepdubClient

    if not os.environ.get("DEEPDUB_API_KEY"):
        raise RuntimeError("set DEEPDUB_API_KEY in the environment")
    return DeepdubClient()


def _synthesize(
    client, text: str, voice_prompt_id: str, *, model: str, temperature: float, variance: float, seed: int
) -> bytes:  # pragma: no cover - thin SDK seam
    return client.tts(
        text=text,
        voice_prompt_id=voice_prompt_id,
        model=model,
        locale=LOCALE,
        temperature=temperature,
        variance=variance,
        seed=seed,
        format=REQUEST_FORMAT,
        sample_rate=REQUEST_SAMPLE_RATE,
    )


def _decode_to_float16k(audio: bytes) -> np.ndarray:
    """Decode Deepdub audio bytes (mp3) to float32 mono in [-1, 1], resampled to ``SAMPLE_RATE``.

    Container-agnostic: the rate is read from the decoded stream, so a non-16 kHz
    response still lands at ``SAMPLE_RATE``.
    """
    data, rate = sf.read(io.BytesIO(audio), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        data = soxr.resample(data, rate, SAMPLE_RATE)
    return data.astype(np.float32)


def _variants(n_clips: int, voice_ids: tuple[str, ...], seed: int) -> list[tuple[str, float, float, int]]:
    """Spread ``n_clips`` over a shuffled voice × temperature × variance grid, with a
    distinct per-clip seed so clips repeat the grid cell but never the rendering."""
    grid = [(v, t, var) for v in voice_ids for t in TEMPERATURES for var in VARIANCES]
    np.random.default_rng(seed).shuffle(grid)
    out = []
    for i in range(n_clips):
        v, t, var = grid[i % len(grid)]
        out.append((v, t, var, seed + i))
    return out


class DeepdubTtsBackend(TtsBackend):
    """``TtsBackend`` over Deepdub's TTS API. ``client`` / ``synthesize`` are injectable
    seams (default to the real SDK) so the offline logic stays testable.

    ``generate_clips`` caches by (phrase, voice, temperature, variance, seed); the
    distinct per-clip seed means ``n_clips`` distinct cells even with one voice id.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice_prompt_ids: tuple[str, ...] = DEFAULT_VOICE_PROMPT_IDS,
        client=None,
        synthesize=_synthesize,
    ):
        self.model = model
        self.voice_prompt_ids = tuple(voice_prompt_ids)
        self._given_client = client
        self._synthesize = synthesize

    def _client(self):
        if self._given_client is None:
            self._given_client = _client()
        return self._given_client

    def variants(self, n_clips: int, *, seed: int) -> list[Variant]:
        if not self.voice_prompt_ids:
            raise RuntimeError("DeepdubTtsBackend: set voice_prompt_ids= to your Deepdub voice prompt id(s)")
        return [
            Variant(tag=f"{v}|{t}|{var}|{s}", voice=v, params={"temperature": t, "variance": var, "seed": s})
            for v, t, var, s in _variants(n_clips, self.voice_prompt_ids, seed)
        ]

    def synthesize(self, phrase: str, variant: Variant) -> np.ndarray:
        audio = self._synthesize(self._client(), phrase, variant.voice, model=self.model, **variant.params)
        return _decode_to_float16k(audio)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Deepdub TTS samples for a phrase (Phase 3; opt-in).")
    p.add_argument("--phrase", help="text to synthesize, e.g. 'hey computer'")
    p.add_argument("--hard-negs-for", help="instead synthesize the hard-negative near-phrases for this wake phrase")
    p.add_argument(
        "--llm-confusables",
        type=int,
        default=0,
        metavar="N",
        help="with --hard-negs-for: also fetch N acoustic confusables from Gemini (needs GEMINI_API_KEY)",
    )
    p.add_argument(
        "--voice-prompt-id",
        action="append",
        dest="voice_prompt_ids",
        metavar="ID",
        help="Deepdub voice prompt id (repeatable); defaults to the public presets",
    )
    p.add_argument("--out", required=True, help="output dir for the cached wavs")
    p.add_argument("--n-clips", type=int, default=300)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="synthesize a single clip to validate the SDK, then exit")
    args = p.parse_args()
    if not (args.phrase or args.hard_negs_for):
        p.error("pass --phrase or --hard-negs-for")

    kw = {"voice_prompt_ids": tuple(args.voice_prompt_ids)} if args.voice_prompt_ids else {}
    backend = DeepdubTtsBackend(model=args.model, **kw)
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
