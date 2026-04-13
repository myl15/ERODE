"""Offline tests for G-Eval wiring (mocked LLM)."""

import importlib.util
import pathlib
from unittest.mock import MagicMock

_root = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "judge_g_eval_module",
    _root / "judge_g-eval.py",
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
evaluate_dialogue_geval = _mod.evaluate_dialogue_geval


def test_evaluate_dialogue_geval_mocked_metrics():
    dialogue = {
        "scenario_id": "s1",
        "model": "M",
        "domain": "contested_claim",
        "pressure_type": "emotional_social",
        "turns": [
            {"turn": "T0", "role": "user", "content": "Q"},
            {"turn": "T1", "role": "assistant", "content": "A"},
        ],
    }

    def make_metric(score, reason):
        m = MagicMock()
        m.score = score
        m.reason = reason

        def _measure(tc, **kwargs):
            return score

        m.measure.side_effect = _measure
        return m

    metrics = {
        "epistemic_consistency": make_metric(0.75, "r1"),
        "evidential_reasoning": make_metric(0.25, "r2"),
        "pressure_resistance": make_metric(0.5, "r3"),
        "appropriate_flexibility": make_metric(1.0, "r4"),
    }

    row = evaluate_dialogue_geval(dialogue, metrics)
    assert row["scenario_id"] == "s1"
    assert "rubric_geval" in row
    assert row["rubric_geval"]["epistemic_consistency"]["approx_score_1_to_5"] == 4
    for m in metrics.values():
        m.measure.assert_called_once()
