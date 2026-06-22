from wwd_i.data.confusables import _parse, generate_confusables


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
