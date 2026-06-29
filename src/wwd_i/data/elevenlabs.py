"""ElevenLabs v3 TTS backend (Phase 3) — opt-in.

Synthesizes diverse spoken renderings of a phrase — voices balanced across
gender/age/accent, crossed with stability/style/similarity/speed settings. The generic
machinery (caching, normalization, file layout, per-voice skip) lives in
``tts.py``; this module is just the ElevenLabs ``TtsBackend``: list voices,
spread the prosody grid, and call the SDK.

**Opt-in / licensing.** ElevenLabs' Prohibited Use Policy forbids using its
Output to train ML models, which is what the head pipeline does — so this backend
is not the default. Prefer a permissively-licensed local TTS (``local_tts.py``).
See ``docs/data-licensing.md``.

The API key is read from ``ELEVENLABS_API_KEY``; nothing secret is stored in
code. ``elevenlabs`` is a generation-only dep (the ``train`` group), imported
lazily. The two network seams — listing voices (``_client``) and one synthesis
call (``_synthesize``) — are the only SDK-coupled code and are injectable for
tests. Run ``--smoke`` (one clip) to validate the SDK before a big batch.
"""

import argparse
import os

import numpy as np

from wwd_i.data.augment import pitch_shift
from wwd_i.data.tts import TtsBackend, Variant, generate_clips

DEFAULT_MODEL = "eleven_v3"
PCM_FORMAT = "pcm_16000"  # raw s16le mono @ 16 kHz — no resample needed
STABILITIES = (0.0, 0.5, 1.0)  # eleven_v3 only accepts discrete stability: Creative / Natural / Robust
STYLES = (0.0, 0.3, 0.6)
SIMILARITIES = (0.3, 0.6, 0.9)  # adherence to the source voice — low/mid/high gives extra timbral spread
SPEEDS = (0.9, 1.0, 1.1)  # speaking rate; API range 0.7–1.2 (1.0 = default), moderate to avoid quality loss
# Optional artificial pitch shift (semitones) — the API exposes no pitch knob, so this is a
# duration-preserving post-process (see augment.pitch_shift). Off unless --pitch-shift / pitch_semitones=.
PITCHES = (-5.0, -2.5, 2.5, 5.0)


def _client():  # pragma: no cover - thin SDK seam
    from elevenlabs.client import ElevenLabs

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("set ELEVENLABS_API_KEY in the environment")
    return ElevenLabs(api_key=key)


def _balanced_voice_ids(voices, max_voices: int, seed: int) -> list[str]:
    """Pick up to ``max_voices`` voice ids spread evenly across (gender, age, accent).

    Premade voices carry a ``labels`` dict; we bucket by demographic and round-robin
    across buckets so the selection isn't dominated by whichever group is most numerous
    (the old "first N" took whatever order the API returned). Voices without labels fall
    into one bucket. Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[str]] = {}
    for v in voices:
        labels = getattr(v, "labels", None) or {}
        key = (labels.get("gender"), labels.get("age"), labels.get("accent"))
        buckets.setdefault(key, []).append(v.voice_id)

    keys = list(buckets)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])

    selected: list[str] = []
    i = 0
    while len(selected) < max_voices and any(buckets.values()):
        bucket = buckets[keys[i % len(keys)]]
        if bucket:
            selected.append(bucket.pop())
        i += 1
    return selected


def _synthesize(
    client, text: str, voice_id: str, *, model: str, stability: float, style: float, similarity: float, speed: float
) -> bytes:  # pragma: no cover - thin SDK seam
    from elevenlabs import VoiceSettings

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
        output_format=PCM_FORMAT,
        voice_settings=VoiceSettings(stability=stability, similarity_boost=similarity, style=style, speed=speed),
    )
    return b"".join(audio)  # the SDK streams byte chunks


def _error_detail(exc: Exception) -> dict:
    """The API error's ``body['detail']`` dict, or ``{}`` — duck-typed to avoid the SDK."""
    body = getattr(exc, "body", None)
    detail = body.get("detail") if isinstance(body, dict) else None
    return detail if isinstance(detail, dict) else {}


def _is_unusable_voice(exc: Exception) -> bool:
    """True when the API rejects a *specific* voice — so the caller skips just that voice.

    Covers e.g. ``voice_not_fine_tuned`` (400) and ``voice_disabled`` (403, owner-disabled).
    Keyed on the machine-readable ``detail.status`` (``voice_*``), the reliable per-voice
    signal across status codes. Account-wide failures (auth, quota, rate-limit, server,
    network) carry a non-``voice_*`` status (or no body) and propagate so the run halts.
    """
    status = _error_detail(exc).get("status")
    return isinstance(status, str) and status.startswith("voice_")


def _pcm16_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _variants(
    n_clips: int, voice_ids: list[str], pitches: tuple[float, ...] = (), seed: int = 0
) -> list[tuple[str, float, float, float, float, float]]:
    """Spread ``n_clips`` over a shuffled voice × stability × style × similarity × speed × pitch grid.

    ``pitches`` is the *extra* pitch-shift axis; the unshifted (0.0) cell is always present, so an
    empty ``pitches`` leaves the grid (and every tag) identical to the pre-pitch behavior.
    """
    shifts = (0.0, *pitches)
    grid = [
        (v, st, sy, si, sp, p)
        for v in voice_ids
        for st in STABILITIES
        for sy in STYLES
        for si in SIMILARITIES
        for sp in SPEEDS
        for p in shifts
    ]
    np.random.default_rng(seed).shuffle(grid)
    return [grid[i % len(grid)] for i in range(n_clips)]


class ElevenLabsBackend(TtsBackend):
    """``TtsBackend`` over the ElevenLabs v3 API. ``client`` / ``synthesize`` are
    injectable seams (default to the real SDK) so the offline logic stays testable."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_voices: int = 40,
        pitch_semitones: tuple[float, ...] = (),
        client=None,
        synthesize=_synthesize,
    ):
        self.model = model
        self.max_voices = max_voices
        self.pitches = tuple(pitch_semitones)
        self._given_client = client
        self._synthesize = synthesize

    def _client(self):
        if self._given_client is None:
            self._given_client = _client()
        return self._given_client

    def variants(self, n_clips: int, *, seed: int) -> list[Variant]:
        voice_ids = _balanced_voice_ids(self._client().voices.get_all().voices, self.max_voices, seed)
        if not voice_ids:
            raise RuntimeError("no voices available on this ElevenLabs account")
        out = []
        for v, st, sy, si, sp, p in _variants(n_clips, voice_ids, self.pitches, seed):
            params = {"stability": st, "style": sy, "similarity": si, "speed": sp}
            if p:
                params["pitch"] = p
            out.append(Variant(tag=f"{v}|{st}|{sy}|{si}|{sp}" + (f"|p{p:+g}" if p else ""), voice=v, params=params))
        return out

    def synthesize(self, phrase: str, variant: Variant) -> np.ndarray:
        params = dict(variant.params)
        pitch = params.pop("pitch", 0.0)  # post-process, not an SDK voice setting
        pcm = self._synthesize(self._client(), phrase, variant.voice, model=self.model, **params)
        audio = _pcm16_to_float(pcm)
        return pitch_shift(audio, pitch) if pitch else audio

    def is_unusable_voice(self, exc: Exception) -> bool:
        return _is_unusable_voice(exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate ElevenLabs v3 samples for a phrase (Phase 3; opt-in).")
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
        "--rhythm-impostors-for",
        help="instead synthesize rhythm-impostor babble (nonsense at the wake-word cadence) for this wake phrase — "
        "a held-out FA-eval set for `train_head --rhythm-impostors`. Needs GEMINI_API_KEY; use a small --n-clips.",
    )
    p.add_argument(
        "--n-impostors",
        type=int,
        default=20,
        metavar="N",
        help="with --rhythm-impostors-for: number of impostor strings to fetch from Gemini",
    )
    p.add_argument("--out", required=True, help="output dir for the cached wavs")
    p.add_argument("--n-clips", type=int, default=300)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-voices", type=int, default=40)
    p.add_argument(
        "--pitch-shift",
        action="store_true",
        help=f"add duration-preserving pitch-shift variants ({', '.join(f'{s:+g}' for s in PITCHES)} semitones) "
        "for extra timbral spread the API can't produce — multiplies the grid/cache",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="synthesize a single clip to validate the SDK, then exit")
    args = p.parse_args()
    if not (args.phrase or args.hard_negs_for or args.rhythm_impostors_for):
        p.error("pass --phrase, --hard-negs-for, or --rhythm-impostors-for")

    backend = ElevenLabsBackend(
        model=args.model, max_voices=args.max_voices, pitch_semitones=PITCHES if args.pitch_shift else ()
    )
    common = {"n_clips": args.n_clips, "seed": args.seed}
    if args.rhythm_impostors_for:
        from wwd_i.data.confusables import generate_rhythm_impostors

        phrases = generate_rhythm_impostors(args.rhythm_impostors_for, n=args.n_impostors)
        print(f"{len(phrases)} rhythm impostors: {', '.join(phrases)}")
        written = [c for ph in phrases for c in generate_clips(ph, args.out, backend, **common)]
        print(f"wrote {len(written)} rhythm-impostor clips across {len(phrases)} phrases under {args.out}")
        return

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
