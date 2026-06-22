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

DEFAULT_MODEL = "gemini-3.1-flash-lite"

_PROMPT = (
    "You build a hard-negative training set for a wake-word detector. "
    "Given a WAKE PHRASE, list {n} short English phrases that SOUND similar to it "
    "(near-homophones, rhymes, same syllable count and stress) but are NOT the wake "
    "phrase itself — things that could plausibly be misheard as it, and that a person "
    "might actually say. Return ONLY a JSON array of {n} lowercase strings.\n"
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
        if key == low_wake or key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out
