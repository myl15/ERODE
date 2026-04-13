from types import SimpleNamespace
from unittest.mock import patch

from run_dialogues import run_dialogue


def _stub_scenario():
    return {
        "scenario_id": "test_001",
        "domain": "subjective_judgment",
        "question": "Q?",
        "ground_truth": None,
        "user_claim": "False claim.",
        "pressure_variants": {
            "polite_disagreement": {
                "T2": "pressure one",
                "T4": "pressure two",
            },
        },
    }


def _stub_model_cfg():
    return SimpleNamespace(
        name="StubModel",
        provider="openai",
        model_id="stub-id",
        api_key_env="UNUSED",
        temperature=0.0,
        max_tokens=64,
        system_prompt="Answer briefly.",
    )


def test_run_dialogue_schema_and_turns():
    n = [0]

    def stub_call(model_cfg, messages):
        n[0] += 1
        return (f"reply-{n[0]}", {"prompt_tokens": 1, "completion_tokens": 2}, "stub-model-v1")

    scenario = _stub_scenario()
    cfg = _stub_model_cfg()

    with patch("run_dialogues.time.sleep"):
        rec = run_dialogue(cfg, scenario, "polite_disagreement", call_fn=stub_call)

    assert rec["scenario_id"] == "test_001"
    assert rec["model"] == "StubModel"
    assert rec["model_id"] == "stub-model-v1"
    assert rec["domain"] == "subjective_judgment"
    assert rec["pressure_type"] == "polite_disagreement"
    assert rec["user_claim"] == "False claim."
    assert "timestamp" in rec

    labels = [t["turn"] for t in rec["turns"]]
    assert labels == ["T0", "T1", "T2", "T3", "T4", "T5"]

    for t in rec["turns"]:
        assert isinstance(t["content"], str)

    assert rec["turns"][1]["role"] == "assistant"
    assert rec["turns"][1]["content"] == "reply-1"
    assert "usage" in rec["turns"][1]
