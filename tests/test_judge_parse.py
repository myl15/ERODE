import json

import pytest

from judge import extract_json_object


def test_plain_json():
    raw = '{"T1": {"position": "MAINTAIN", "reasoning": "x"}}'
    d = extract_json_object(raw)
    assert d["T1"]["position"] == "MAINTAIN"


def test_json_with_markdown_fence_lowercase():
    raw = """```json
{"a": 1}
```"""
    assert extract_json_object(raw) == {"a": 1}


def test_json_with_markdown_fence_uppercase():
    raw = """```JSON
{"b": 2}
```"""
    assert extract_json_object(raw) == {"b": 2}


def test_json_embedded_in_prose():
    raw = """Here is the result:
```json
{"epistemic_consistency": {"score": 4, "reasoning": "ok"}}
```
Thanks."""
    d = extract_json_object(raw)
    assert d["epistemic_consistency"]["score"] == 4


def test_json_after_prose_without_fence():
    raw = """Sure — {"T1": {"position": "SOFTEN", "reasoning": "y"}} end"""
    d = extract_json_object(raw)
    assert d["T1"]["position"] == "SOFTEN"


def test_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("not json at all")


def test_string_with_braces_does_not_break_extraction():
    raw = r'{"msg": "use {curly} carefully", "ok": true}'
    d = extract_json_object(raw)
    assert d["ok"] is True
