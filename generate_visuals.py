"""
Build metrics summary and all figures from transcripts → features → judgments.

Default: single OpenAI run (model key gpt4o). Pass more keys for comparisons.

Usage:
  python generate_visuals.py
  python generate_visuals.py gpt4o
  python generate_visuals.py gpt4o claude35sonnet gemini25flash

Requires, per model key:
  features/<key>_features.jsonl
  judgments/<key>_judgments.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from config import ANALYSIS_DIR, MODELS
from metrics import compute_all_metrics
from stats_tests import compute_significance_summary
from analyze import (
    load_features_list,
    load_judgments_list,
    plot_domain_pressure_heatmaps,
    plot_ers_radar,
    plot_heuristic_judge_confusion,
    plot_hedge_trajectory,
    plot_hedge_trajectory_by_domain,
    plot_lexical_trajectory_by_domain,
    plot_position_transition_matrices,
    plot_pressure_response_curve,
    plot_rubric_by_domain,
    plot_rubric_by_pressure,
    plot_sycophancy_by_domain,
    plot_sycophancy_by_pressure,
    plot_t5_capitulation_given_t3,
)


def _require_files(model_key: str) -> list[str]:
    """Return list of missing paths (empty if ok)."""
    from config import FEATURES_DIR, JUDGMENTS_DIR

    missing = []
    feat = os.path.join(FEATURES_DIR, f"{model_key}_features.jsonl")
    judg = os.path.join(JUDGMENTS_DIR, f"{model_key}_judgments.jsonl")
    if not os.path.isfile(feat):
        missing.append(feat)
    if not os.path.isfile(judg):
        missing.append(judg)
    return missing


def _model_labels(model_keys: list[str]) -> dict[str, str]:
    return {mk: MODELS[mk].name for mk in model_keys if mk in MODELS}


def main(model_keys: list[str]) -> int:
    for mk in model_keys:
        miss = _require_files(mk)
        if miss:
            print(f"Missing inputs for {mk!r}:", file=sys.stderr)
            for p in miss:
                print(f"  {p}", file=sys.stderr)
            return 1

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    labels = _model_labels(model_keys)

    print("Computing metrics...")
    summary = compute_all_metrics(model_keys)
    summary_path = os.path.join(ANALYSIS_DIR, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("Computing statistical significance artifacts...")
    significance = compute_significance_summary(
        model_keys=model_keys,
        baseline_model=model_keys[0] if model_keys else None,
    )
    significance_path = os.path.join(ANALYSIS_DIR, "significance_summary.json")
    with open(significance_path, "w") as f:
        json.dump(significance, f, indent=2)
    print(f"Wrote {significance_path}")

    main_path = os.path.join(ANALYSIS_DIR, "significance_main_baseline_vs_others.json")
    with open(main_path, "w") as f:
        json.dump(significance["results"]["main_baseline_vs_others"], f, indent=2)
    print(f"Wrote {main_path}")

    appendix_path = os.path.join(ANALYSIS_DIR, "significance_appendix_pairwise.json")
    with open(appendix_path, "w") as f:
        json.dump(significance["results"]["appendix_pairwise"], f, indent=2)
    print(f"Wrote {appendix_path}")

    judgments_by_model = {mk: load_judgments_list(mk) for mk in model_keys}
    features_by_model = {mk: load_features_list(mk) for mk in model_keys}

    plots = [
        (
            "sycophancy_by_domain.png",
            lambda: plot_sycophancy_by_domain(
                summary, model_keys,
                os.path.join(ANALYSIS_DIR, "sycophancy_by_domain.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "sycophancy_by_pressure.png",
            lambda: plot_sycophancy_by_pressure(
                summary, model_keys,
                os.path.join(ANALYSIS_DIR, "sycophancy_by_pressure.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "hedge_trajectory.png",
            lambda: plot_hedge_trajectory(
                features_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "hedge_trajectory.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "pressure_response_curve.png",
            lambda: plot_pressure_response_curve(
                judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "pressure_response_curve.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "ers_radar.png",
            lambda: plot_ers_radar(
                summary, model_keys,
                os.path.join(ANALYSIS_DIR, "ers_radar.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "domain_pressure_heatmaps.png",
            lambda: plot_domain_pressure_heatmaps(
                summary, model_keys,
                os.path.join(ANALYSIS_DIR, "domain_pressure_heatmaps.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "rubric_by_domain.png",
            lambda: plot_rubric_by_domain(
                judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "rubric_by_domain.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "rubric_by_pressure.png",
            lambda: plot_rubric_by_pressure(
                judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "rubric_by_pressure.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "position_transition_matrices.png",
            lambda: plot_position_transition_matrices(
                judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "position_transition_matrices.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "t5_capitulation_given_t3.png",
            lambda: plot_t5_capitulation_given_t3(
                judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "t5_capitulation_given_t3.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "hedge_trajectory_by_domain.png",
            lambda: plot_hedge_trajectory_by_domain(
                features_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "hedge_trajectory_by_domain.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "lexical_trajectory_by_domain.png",
            lambda: plot_lexical_trajectory_by_domain(
                features_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "lexical_trajectory_by_domain.png"),
                model_labels=labels or None,
            ),
        ),
        (
            "heuristic_judge_confusion_t5.png",
            lambda: plot_heuristic_judge_confusion(
                features_by_model, judgments_by_model, model_keys,
                os.path.join(ANALYSIS_DIR, "heuristic_judge_confusion_t5.png"),
                model_labels=labels or None,
            ),
        ),
    ]

    for name, fn in plots:
        fn()
        print(f"Wrote {os.path.join(ANALYSIS_DIR, name)}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate analysis figures and metrics JSON.")
    p.add_argument(
        "models",
        nargs="*",
        default=["gpt4o", "gemini25flash"],
        help="Model config keys (default: gpt4o gemini25flash). Must match features/judgment filenames.",
    )
    args = p.parse_args()
    raise SystemExit(main(args.models))
