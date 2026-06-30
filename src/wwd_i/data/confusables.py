"""LLM-generated acoustic confusables for a wake phrase (Phase 3).

The fixed ``GENERIC_CONFUSABLES`` in :mod:`wwd_i.data.negatives` are
word-independent filler. The *hardest* negatives are phrases that sound like the
wake phrase itself — near-homophones, rhymes, matching syllable count/stress —
because that is where false triggers live. This asks Gemini for such phrases for
a given wake phrase, to extend the per-word hard-negative set.

``GEMINI_API_KEY`` is read from the environment; ``google-genai`` is a
generation-only dep (the ``train`` group), imported lazily. The single network
seam is ``_generate`` (injectable for tests); parsing/filtering is offline and
unit-tested.
"""

import json
import os
import re

DEFAULT_MODEL = "gemini-3.5-flash"

# Confusables must be CLOSE (same rhythm) yet RELIABLY DISTINGUISHABLE: a phrase
# pronounced identically to the wake word (Oksana/Aksana/Axana) overlaps the
# positive manifold and, used as a negative, teaches the head to reject true
# positives (higher FR). So we forbid homophones / mere respellings and require at
# least one clearly audible phoneme difference, while keeping the syllable count
# and stress so the rhythm is still the confuser.
_PROMPT = (
    "You build a hard-negative training set for a wake-word detector. "
    "Given a WAKE PHRASE, list {n} short phrases, in the SAME language and script as the wake phrase, "
    "that are ACOUSTICALLY CLOSE to it — the same number of syllables and the same stress pattern, so they "
    "share its rhythm — but are CLEARLY DISTINGUISHABLE: each must differ from the wake phrase by at least one "
    "clearly AUDIBLE phoneme (a different consonant or a different vowel), not merely a different spelling. "
    "Do NOT return homophones or phrases pronounced identically to the wake phrase, and do NOT return the wake "
    'phrase merely re-spelled (e.g. for "Oksana" do NOT return "Aksana" or "Axana"). '
    "Prefer real words or names a person might actually say. "
    "Return ONLY a JSON array of {n} lowercase strings.\n"
    'WAKE PHRASE: "{wake}"'
)

# Rhythm impostors are NON-WORD babble that copies only the wake word's cadence
# (syllable count + stress) with unrelated voiced phonemes — the "мя-мя-мЯ fires
# Samara" failure mode. They are guaranteed negatives (no positive-manifold
# overlap) and are used as a held-out FA-eval set to expose rhythm-only triggers.
_RHYTHM_PROMPT = (
    "You build a RHYTHM-IMPOSTOR set to test a wake-word detector for false triggers. "
    "Given a WAKE PHRASE, list {n} short NONSENSE syllable strings, in the same language and script as the wake "
    "phrase, that copy ONLY its RHYTHM — the same number of syllables and the same stressed syllable — but share "
    "NONE of its consonant or vowel identity beyond being voiced. Build them from simple open sonorant syllables "
    "(m/n/l/r/y + a/o/u), write the STRESSED syllable in UPPERCASE, and separate syllables with spaces "
    '(e.g. for a 3-syllable phrase stressed on the 2nd: "na NA na", "lo MU lo", "mya MYA ma"). '
    "They must be NON-WORDS in every language, so they differ from the wake phrase only in rhythm. "
    "Return ONLY a JSON array of {n} strings.\n"
    'WAKE PHRASE: "{wake}"'
)


def _client():  # pragma: no cover - thin SDK seam
    from google import genai

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("set GEMINI_API_KEY in the environment")
    return genai.Client(api_key=key)


def _generate(client, prompt: str, *, model: str) -> str:  # pragma: no cover - thin SDK seam
    from google.genai import types

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=list[str]),
    )
    return resp.text or ""


def _parse(text: str) -> list[str]:
    """Best-effort phrase list from the model reply (JSON array, fenced, or looser)."""
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, list):
                return [s for x in data if (s := str(x).strip())]
        except json.JSONDecodeError:
            pass
    return [s for part in re.split(r"[\n,]", text) if (s := part.strip(" \t\"'-•"))]


def generate_confusables(
    wake_phrase: str,
    *,
    n: int = 20,
    model: str = DEFAULT_MODEL,
    client=None,
    generate=_generate,
) -> list[str]:
    """Ask Gemini for up to ``n`` phrases that sound like ``wake_phrase``.

    Returns phrases de-duplicated case-insensitively, with the wake phrase itself
    removed, in model order. ``client`` / ``generate`` are injectable seams
    (default to the real SDK). Network failures propagate.
    """
    wake = wake_phrase.strip()
    if not wake:
        raise ValueError("empty wake phrase")
    client = client or _client()
    text = generate(client, _PROMPT.format(n=n, wake=wake), model=model)

    low_wake = wake.lower()
    seen: set[str] = set()
    out: list[str] = []
    for phrase in _parse(text):
        key = phrase.lower()
        if key in low_wake or key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def generate_rhythm_impostors(
    wake_phrase: str,
    *,
    n: int = 20,
    model: str = DEFAULT_MODEL,
    client=None,
    generate=_generate,
) -> list[str]:
    """Ask Gemini for up to ``n`` non-word babble strings at ``wake_phrase``'s cadence.

    These copy the wake word's syllable count and stress but none of its phonemes
    (the "мя-мя-мЯ" rhythm-imposter case) — guaranteed negatives for a held-out
    FA-eval set (see ``train.train_head --rhythm-impostors``). Returned in model
    order, de-duplicated case-insensitively but with case preserved (the uppercase
    marks the stressed syllable for the TTS). ``client`` / ``generate`` are the
    same injectable seams as :func:`generate_confusables`.
    """
    wake = wake_phrase.strip()
    if not wake:
        raise ValueError("empty wake phrase")
    client = client or _client()
    text = generate(client, _RHYTHM_PROMPT.format(n=n, wake=wake), model=model)

    seen: set[str] = set()
    out: list[str] = []
    for phrase in _parse(text):
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out
