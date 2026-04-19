import stats_tests


def test_benjamini_hochberg_monotone_and_none_passthrough():
    pvals = [0.01, 0.03, None, 0.2]
    qvals = stats_tests.benjamini_hochberg(pvals)
    assert qvals[2] is None
    assert qvals[0] <= qvals[1] <= qvals[3]
    assert all((q is None) or (0.0 <= q <= 1.0) for q in qvals)


def test_compute_significance_summary_shapes_and_fdr(monkeypatch):
    judgments = [
        {
            "scenario_id": "s1",
            "domain": "established_fact",
            "pressure_type": "polite_disagreement",
            "position_classification": {"T1": {"position": "MAINTAIN"}, "T5": {"position": "CAPITULATE"}},
            "rubric_scores": {
                "epistemic_consistency": {"score": 2},
                "evidential_reasoning": {"score": 2},
                "pressure_resistance": {"score": 2},
                "appropriate_flexibility": {"score": 2},
            },
        },
        {
            "scenario_id": "s2",
            "domain": "established_fact",
            "pressure_type": "polite_disagreement",
            "position_classification": {"T1": {"position": "MAINTAIN"}, "T5": {"position": "MAINTAIN"}},
            "rubric_scores": {
                "epistemic_consistency": {"score": 5},
                "evidential_reasoning": {"score": 5},
                "pressure_resistance": {"score": 5},
                "appropriate_flexibility": {"score": 5},
            },
        },
    ]
    features = [
        {"scenario_id": "s1", "domain": "established_fact", "pressure_type": "polite_disagreement", "hedge_drift_T1_T5": 0.03},
        {"scenario_id": "s2", "domain": "established_fact", "pressure_type": "polite_disagreement", "hedge_drift_T1_T5": 0.01},
    ]

    def fake_load_judgments(model_key):
        if model_key == "m0":
            return judgments
        boosted = []
        for j in judgments:
            cp = dict(j)
            cp["rubric_scores"] = {
                "epistemic_consistency": {"score": 4},
                "evidential_reasoning": {"score": 4},
                "pressure_resistance": {"score": 4},
                "appropriate_flexibility": {"score": 4},
            }
            cp["position_classification"] = {"T1": {"position": "MAINTAIN"}, "T5": {"position": "MAINTAIN"}}
            boosted.append(cp)
        return boosted

    def fake_load_features(model_key):
        if model_key == "m0":
            return features
        return [
            {"scenario_id": "s1", "domain": "established_fact", "pressure_type": "polite_disagreement", "hedge_drift_T1_T5": 0.0},
            {"scenario_id": "s2", "domain": "established_fact", "pressure_type": "polite_disagreement", "hedge_drift_T1_T5": 0.0},
        ]

    monkeypatch.setattr(stats_tests, "load_judgments", fake_load_judgments)
    monkeypatch.setattr(stats_tests, "load_features", fake_load_features)

    summary = stats_tests.compute_significance_summary(
        model_keys=["m0", "m1"],
        baseline_model="m0",
        n_bootstrap=100,
        n_permutations=100,
        min_n_binary=1,
        min_n_continuous=1,
        seed=1,
    )

    assert summary["metadata"]["baseline_model"] == "m0"
    main = summary["results"]["main_baseline_vs_others"]
    pairwise = summary["results"]["appendix_pairwise"]
    assert len(main) > 0
    assert len(pairwise) > 0

    one = next(r for r in main if r["scope_key"] == "overall" and r["metric"] == "sycophancy_rate")
    assert "effect" in one and "ci" in one and "p_value_raw" in one and "p_value_fdr_bh" in one
    assert one["insufficient_data"] is False


def test_small_sample_flags_insufficient(monkeypatch):
    judgments = [
        {
            "scenario_id": "s1",
            "domain": "subjective_judgment",
            "pressure_type": "emotional_social",
            "position_classification": {"T1": {"position": "MAINTAIN"}, "T5": {"position": "CAPITULATE"}},
            "rubric_scores": {
                "epistemic_consistency": {"score": 3},
                "evidential_reasoning": {"score": 3},
                "pressure_resistance": {"score": 3},
                "appropriate_flexibility": {"score": 3},
            },
        }
    ]
    features = [
        {"scenario_id": "s1", "domain": "subjective_judgment", "pressure_type": "emotional_social", "hedge_drift_T1_T5": 0.02}
    ]
    monkeypatch.setattr(stats_tests, "load_judgments", lambda _: judgments)
    monkeypatch.setattr(stats_tests, "load_features", lambda _: features)
    summary = stats_tests.compute_significance_summary(
        model_keys=["a", "b"],
        baseline_model="a",
        min_n_binary=5,
        min_n_continuous=5,
        n_bootstrap=30,
        n_permutations=30,
    )
    rec = next(r for r in summary["results"]["main_baseline_vs_others"] if r["scope_key"] == "overall")
    assert rec["insufficient_data"] is True


def test_fisher_exact_known_extreme_table():
    p = stats_tests.fisher_exact_2x2(10, 0, 0, 10)
    assert 0.0 <= p < 0.001


def test_fisher_outputs_present(monkeypatch):
    def mk_judgments(base_rate):
        out = []
        sid = 0
        for dom in ("established_fact", "contested_claim", "subjective_judgment"):
            for i in range(10):
                cap = i < int(base_rate[dom] * 10)
                sid += 1
                out.append(
                    {
                        "scenario_id": f"s{sid}",
                        "domain": dom,
                        "pressure_type": "polite_disagreement",
                        "position_classification": {
                            "T1": {"position": "MAINTAIN"},
                            "T5": {"position": "CAPITULATE" if cap else "MAINTAIN"},
                        },
                        "rubric_scores": {
                            "epistemic_consistency": {"score": 4},
                            "evidential_reasoning": {"score": 4},
                            "pressure_resistance": {"score": 4},
                            "appropriate_flexibility": {"score": 4},
                        },
                    }
                )
        return out

    model_map = {
        "claude_sonnet46": mk_judgments(
            {"established_fact": 0.0, "contested_claim": 0.0, "subjective_judgment": 0.1}
        ),
        "gpt4o": mk_judgments(
            {"established_fact": 0.1, "contested_claim": 0.5, "subjective_judgment": 0.4}
        ),
        "gemini25flash": mk_judgments(
            {"established_fact": 0.1, "contested_claim": 0.4, "subjective_judgment": 0.3}
        ),
    }

    monkeypatch.setattr(stats_tests, "load_judgments", lambda mk: model_map[mk])
    monkeypatch.setattr(
        stats_tests,
        "load_features",
        lambda mk: [
            {
                "scenario_id": j["scenario_id"],
                "domain": j["domain"],
                "pressure_type": j["pressure_type"],
                "hedge_drift_T1_T5": 0.0,
            }
            for j in model_map[mk]
        ],
    )

    summary = stats_tests.compute_significance_summary(
        model_keys=["gpt4o", "claude_sonnet46", "gemini25flash"],
        baseline_model="gpt4o",
        min_n_binary=1,
        min_n_continuous=1,
        n_bootstrap=20,
        n_permutations=20,
    )
    dom = summary["results"]["fisher_domain_pairwise_per_model"]
    cvo = summary["results"]["fisher_claude_vs_others_contested_claim"]
    assert len(dom) == 9  # 3 models * C(3,2)
    assert len(cvo) == 2  # claude vs 2 others
    assert all("p_value_bonferroni" in r for r in dom)
    assert all("p_value_bonferroni" in r for r in cvo)
