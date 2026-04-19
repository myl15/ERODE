import json

import pytest

from judge import extract_json_object
from judge import _partition_judgment_records_for_resume


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


def test_partition_judgment_records_for_resume_reruns_failed_only_without_success():
    existing = [
        {
            "scenario_id": "s1",
            "pressure_type": "p1",
            "position_classification": {"error": "Failed to parse", "raw": "x"},
        },
        {
            "scenario_id": "s2",
            "pressure_type": "p2",
            "position_classification": {"T1": {"position": "MAINTAIN"}},
        },
    ]

    keep_records, completed, rerun_pairs = _partition_judgment_records_for_resume(existing)

    assert ("s1", "p1") in rerun_pairs
    assert ("s1", "p1") not in completed
    assert ("s2", "p2") in completed
    assert ("s2", "p2") not in rerun_pairs
    assert len(keep_records) == 1
    assert keep_records[0]["scenario_id"] == "s2"


def test_partition_judgment_records_for_resume_does_not_rerun_if_success_exists():
    existing = [
        {
            "scenario_id": "s3",
            "pressure_type": "p3",
            "position_classification": {"error": "Failed to parse", "raw": "x"},
        },
        {
            "scenario_id": "s3",
            "pressure_type": "p3",
            "position_classification": {"T1": {"position": "SOFTEN"}},
        },
    ]

    keep_records, completed, rerun_pairs = _partition_judgment_records_for_resume(existing)

    assert ("s3", "p3") in completed
    assert ("s3", "p3") not in rerun_pairs
    assert len(keep_records) == 1
    assert keep_records[0]["position_classification"]["T1"]["position"] == "SOFTEN"


def test_partition_judgment_records_for_resume_keeps_non_failed_duplicates():
    existing = [
        {
            "scenario_id": "s4",
            "pressure_type": "p4",
            "position_classification": {"T1": {"position": "MAINTAIN"}},
        },
        {
            "scenario_id": "s4",
            "pressure_type": "p4",
            "position_classification": {"T1": {"position": "MAINTAIN"}},
        },
    ]

    keep_records, completed, rerun_pairs = _partition_judgment_records_for_resume(existing)

    assert ("s4", "p4") in completed
    assert ("s4", "p4") not in rerun_pairs
    assert len(keep_records) == 2
