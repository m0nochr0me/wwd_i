from wwd_i.data.confusables import (
    _PROMPT,
    DEFAULT_MODEL,
    _parse,
    generate_confusables,
    generate_rhythm_impostors,
)


def test_parse_json_array():
    assert _parse('["maria", "all year"]') == ["maria", "all year"]


def test_parse_strips_markdown_fence():
    assert _parse('```json\n["maria", "all year"]\n```') == ["maria", "all year"]


def test_parse_fallback_lines():
    assert _parse("maria\nall year\n- olalia") == ["maria", "all year", "olalia"]


def test_generate_confusables_dedupes_and_drops_wake():
    out = generate_confusables(
        "Aliyah",
        n=5,
        client=object(),  # truthy -> real _client()/SDK never touched
        generate=lambda c, prompt, *, model: '["Maria", "maria", "Aliyah", "all year"]',
    )
    assert out == ["Maria", "all year"]  # case-insensitive dedup, wake phrase dropped


def test_generate_confusables_passes_n_and_wake_into_prompt():
    seen = {}

    def fake(c, prompt, *, model):
        seen["prompt"] = prompt
        return "[]"

    generate_confusables("hey computer", n=7, client=object(), generate=fake)
    assert "hey computer" in seen["prompt"] and "7" in seen["prompt"]


def test_generate_confusables_empty_wake_raises():
    try:
        generate_confusables("   ", client=object(), generate=lambda *a, **k: "[]")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty wake phrase")


def test_default_model():
    assert DEFAULT_MODEL == "gemini-3.5-flash"


def test_prompt_forbids_identical_sounding_but_keeps_rhythm():
    p = _PROMPT.lower()
    assert "homophone" in p  # must not return phrases pronounced identically
    assert "audible phoneme" in p  # must differ by a real, audible phoneme
    assert "syllable" in p and "stress" in p  # still rhythm-matched


def test_generate_rhythm_impostors_dedupes_preserving_case():
    out = generate_rhythm_impostors(
        "Samara",
        n=4,
        client=object(),  # truthy -> real _client()/SDK never touched
        generate=lambda c, prompt, *, model: '["na NA na", "lo MU lo", "na NA na", "RA ri ru"]',
    )
    assert out == ["na NA na", "lo MU lo", "RA ri ru"]  # case-insensitive dedup, case (stress) preserved


def test_generate_rhythm_impostors_passes_n_and_wake_into_prompt():
    seen = {}

    def fake(c, prompt, *, model):
        seen["prompt"] = prompt
        return "[]"

    generate_rhythm_impostors("Samara", n=6, client=object(), generate=fake)
    assert "Samara" in seen["prompt"] and "6" in seen["prompt"]


def test_generate_rhythm_impostors_empty_wake_raises():
    try:
        generate_rhythm_impostors("   ", client=object(), generate=lambda *a, **k: "[]")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty wake phrase")
