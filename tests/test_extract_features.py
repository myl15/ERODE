from extract_features import extract_dialogue_features


def test_extract_dialogue_features_minimal_transcript():
    dialogue = {
        "scenario_id": "x",
        "model": "M",
        "domain": "established_fact",
        "pressure_type": "polite_disagreement",
        "ground_truth": "truth",
        "user_claim": "lie",
        "turns": [
            {"turn": "T0", "role": "user", "content": "Question?"},
            {"turn": "T1", "role": "assistant", "content": "I think the answer is clear and obvious."},
            {"turn": "T2", "role": "user", "content": "push"},
            {"turn": "T3", "role": "assistant", "content": "I still believe that however you may see it differently."},
            {"turn": "T4", "role": "user", "content": "push2"},
            {"turn": "T5", "role": "assistant", "content": "Perhaps you are right about some aspects."},
        ],
    }
    feats = extract_dialogue_features(dialogue)
    assert "T1" in feats and "T3" in feats and "T5" in feats
    assert feats["T1"]["hedge_density"] >= 0
    assert "hedge_drift_T1_T5" in feats
