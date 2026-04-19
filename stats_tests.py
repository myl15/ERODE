"""
Statistical significance routines for SycoBench judge-output comparisons.

Primary reporting goal:
- Effect sizes + confidence intervals (main evidence)
- Multiplicity-corrected p-values (secondary evidence)
"""

from __future__ import annotations

import itertools
import json
import math
import os
from typing import Callable

import numpy as np

from config import FEATURES_DIR, JUDGMENTS_DIR
from metrics import DOMAINS, PRESSURE_TYPES, ers_for_judgment, sycophancy_breakdown

METRIC_KEYS = ("sycophancy_rate", "ERS", "hedge_drift")


def load_judgments(model_key: str) -> list[dict]:
    path = os.path.join(JUDGMENTS_DIR, f"{model_key}_judgments.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_features(model_key: str) -> list[dict]:
    path = os.path.join(FEATURES_DIR, f"{model_key}_features.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]


def benjamini_hochberg(pvals: list[float | None]) -> list[float | None]:
    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not indexed:
        return [None for _ in pvals]
    m = len(indexed)
    indexed.sort(key=lambda x: x[1])
    adjusted_sorted = [0.0] * m
    prev = 1.0
    for k in range(m - 1, -1, -1):
        rank = k + 1
        raw = indexed[k][1] * m / rank
        adj = min(prev, raw)
        adjusted_sorted[k] = min(1.0, adj)
        prev = adjusted_sorted[k]
    out = [None for _ in pvals]
    for (orig_idx, _), adj in zip(indexed, adjusted_sorted):
        out[orig_idx] = adj
    return out


def bonferroni_correction(pvals: list[float | None]) -> list[float | None]:
    valid = [p for p in pvals if p is not None]
    m = len(valid)
    if m == 0:
        return [None for _ in pvals]
    return [None if p is None else min(1.0, p * m) for p in pvals]


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_pmf(x: int, r1: int, r2: int, c1: int) -> float:
    n = r1 + r2
    logp = _log_choose(r1, x) + _log_choose(r2, c1 - x) - _log_choose(n, c1)
    return float(math.exp(logp))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact p-value for table:
      [[a, b],
       [c, d]]
    """
    r1, r2 = a + b, c + d
    c1 = a + c
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_obs = _hypergeom_pmf(a, r1, r2, c1)
    p_total = 0.0
    for x in range(lo, hi + 1):
        px = _hypergeom_pmf(x, r1, r2, c1)
        if px <= p_obs + 1e-15:
            p_total += px
    return min(1.0, p_total)


def _bootstrap_ci(
    rng: np.random.Generator,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int,
    ci_level: float,
) -> tuple[float, float]:
    sims = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sa = rng.choice(arr_a, size=len(arr_a), replace=True)
        sb = rng.choice(arr_b, size=len(arr_b), replace=True)
        sims[i] = stat_fn(sa, sb)
    alpha = 1.0 - ci_level
    low = float(np.quantile(sims, alpha / 2.0))
    high = float(np.quantile(sims, 1.0 - alpha / 2.0))
    return low, high


def _permutation_pvalue(
    rng: np.random.Generator,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
) -> float:
    observed = abs(stat_fn(arr_a, arr_b))
    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)
    hits = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined)
        sa = perm[:n_a]
        sb = perm[n_a:]
        if abs(stat_fn(sa, sb)) >= observed:
            hits += 1
    return (hits + 1.0) / (n_permutations + 1.0)


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def _safe_log_odds_ratio(a_yes: int, a_no: int, b_yes: int, b_no: int) -> float:
    # Haldane-Anscombe correction for zero-cell robustness.
    aa = a_yes + 0.5
    ab = a_no + 0.5
    ba = b_yes + 0.5
    bb = b_no + 0.5
    return float(np.log((aa * bb) / (ab * ba)))


def _binary_test(
    rng: np.random.Generator,
    a_success: int,
    a_total: int,
    b_success: int,
    b_total: int,
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
) -> dict:
    a_fail = a_total - a_success
    b_fail = b_total - b_success
    a = np.array([1] * a_success + [0] * a_fail, dtype=float)
    b = np.array([1] * b_success + [0] * b_fail, dtype=float)
    rd = float(np.mean(a) - np.mean(b))
    lor = _safe_log_odds_ratio(a_success, a_fail, b_success, b_fail)
    ci_low, ci_high = _bootstrap_ci(
        rng=rng,
        arr_a=a,
        arr_b=b,
        stat_fn=lambda x, y: float(np.mean(x) - np.mean(y)),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )
    pval = _permutation_pvalue(
        rng=rng,
        arr_a=a,
        arr_b=b,
        stat_fn=lambda x, y: float(np.mean(x) - np.mean(y)),
        n_permutations=n_permutations,
    )
    return {
        "effect": {
            "risk_difference": rd,
            "log_odds_ratio": lor,
        },
        "ci": {
            "risk_difference": [ci_low, ci_high],
            "level": ci_level,
        },
        "p_value_raw": pval,
    }


def _continuous_test(
    rng: np.random.Generator,
    arr_a: list[float],
    arr_b: list[float],
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
) -> dict:
    a = np.array(arr_a, dtype=float)
    b = np.array(arr_b, dtype=float)
    mean_diff = float(np.mean(a) - np.mean(b))
    d = _cohen_d(a, b)
    ci_low, ci_high = _bootstrap_ci(
        rng=rng,
        arr_a=a,
        arr_b=b,
        stat_fn=lambda x, y: float(np.mean(x) - np.mean(y)),
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
    )
    pval = _permutation_pvalue(
        rng=rng,
        arr_a=a,
        arr_b=b,
        stat_fn=lambda x, y: float(np.mean(x) - np.mean(y)),
        n_permutations=n_permutations,
    )
    return {
        "effect": {
            "mean_difference": mean_diff,
            "cohen_d": d,
        },
        "ci": {
            "mean_difference": [ci_low, ci_high],
            "level": ci_level,
        },
        "p_value_raw": pval,
    }


def _scenario_filters() -> list[tuple[str, str, Callable[[dict], bool]]]:
    filters = [("overall", "overall", lambda j: True)]
    filters.extend((f"domain::{d}", d, lambda j, d=d: j.get("domain") == d) for d in DOMAINS)
    filters.extend(
        (f"pressure::{p}", p, lambda j, p=p: j.get("pressure_type") == p) for p in PRESSURE_TYPES
    )
    filters.extend(
        (
            f"domain_pressure::{d}::{p}",
            f"{d}::{p}",
            lambda j, d=d, p=p: j.get("domain") == d and j.get("pressure_type") == p,
        )
        for d in DOMAINS
        for p in PRESSURE_TYPES
    )
    return filters


def _collect_metric_data(
    judgments: list[dict],
    features: list[dict],
    where: Callable[[dict], bool],
) -> dict[str, dict]:
    jsub = [j for j in judgments if where(j)]
    sid_to_feat = {f.get("scenario_id"): f for f in features}

    eligible, sycophantic, _ = sycophancy_breakdown(jsub)
    ers_values = [e for e in (ers_for_judgment(j) for j in jsub) if e is not None]
    hedge_values = []
    for j in jsub:
        f = sid_to_feat.get(j.get("scenario_id"))
        if isinstance(f, dict) and "hedge_drift_T1_T5" in f:
            hedge_values.append(float(f["hedge_drift_T1_T5"]))
    return {
        "sycophancy_rate": {"success": sycophantic, "total": eligible},
        "ERS": {"values": ers_values},
        "hedge_drift": {"values": hedge_values},
    }


def _compare_metric(
    rng: np.random.Generator,
    metric_key: str,
    a_data: dict,
    b_data: dict,
    n_bootstrap: int,
    n_permutations: int,
    ci_level: float,
    min_n: int,
) -> dict:
    if metric_key == "sycophancy_rate":
        at = int(a_data["total"])
        bt = int(b_data["total"])
        if at < min_n or bt < min_n:
            return {
                "metric": metric_key,
                "insufficient_data": True,
                "minimum_n_required": min_n,
                "n_a": at,
                "n_b": bt,
            }
        asu = int(a_data["success"])
        bsu = int(b_data["success"])
        out = _binary_test(
            rng=rng,
            a_success=asu,
            a_total=at,
            b_success=bsu,
            b_total=bt,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            ci_level=ci_level,
        )
        out["n_a"] = at
        out["n_b"] = bt
        out["rate_a"] = asu / at if at else None
        out["rate_b"] = bsu / bt if bt else None
        out["insufficient_data"] = False
        out["metric"] = metric_key
        return out

    arr_a = list(a_data["values"])
    arr_b = list(b_data["values"])
    if len(arr_a) < min_n or len(arr_b) < min_n:
        return {
            "metric": metric_key,
            "insufficient_data": True,
            "minimum_n_required": min_n,
            "n_a": len(arr_a),
            "n_b": len(arr_b),
        }
    out = _continuous_test(
        rng=rng,
        arr_a=arr_a,
        arr_b=arr_b,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        ci_level=ci_level,
    )
    out["n_a"] = len(arr_a)
    out["n_b"] = len(arr_b)
    out["mean_a"] = float(np.mean(arr_a))
    out["mean_b"] = float(np.mean(arr_b))
    out["insufficient_data"] = False
    out["metric"] = metric_key
    return out


def _apply_fdr(records: list[dict]) -> None:
    pvals = [r.get("p_value_raw") if not r.get("insufficient_data") else None for r in records]
    adj = benjamini_hochberg(pvals)
    for r, q in zip(records, adj):
        r["p_value_fdr_bh"] = q


def _fisher_domain_pairwise_per_model(judgments_by_model: dict[str, list[dict]]) -> list[dict]:
    records = []
    for mk, judgments in judgments_by_model.items():
        for d1, d2 in itertools.combinations(DOMAINS, 2):
            n1, s1, _ = sycophancy_breakdown(judgments, domain_filter=d1)
            n2, s2, _ = sycophancy_breakdown(judgments, domain_filter=d2)
            f1 = n1 - s1
            f2 = n2 - s2
            if n1 == 0 or n2 == 0:
                rec = {
                    "model": mk,
                    "domain_a": d1,
                    "domain_b": d2,
                    "n_a": n1,
                    "n_b": n2,
                    "success_a": s1,
                    "success_b": s2,
                    "insufficient_data": True,
                    "p_value_raw": None,
                }
            else:
                rec = {
                    "model": mk,
                    "domain_a": d1,
                    "domain_b": d2,
                    "n_a": n1,
                    "n_b": n2,
                    "success_a": s1,
                    "success_b": s2,
                    "rate_a": s1 / n1,
                    "rate_b": s2 / n2,
                    "insufficient_data": False,
                    "p_value_raw": fisher_exact_2x2(s1, f1, s2, f2),
                }
            records.append(rec)
    adjusted = bonferroni_correction([r["p_value_raw"] for r in records])
    for r, p_adj in zip(records, adjusted):
        r["p_value_bonferroni"] = p_adj
    return records


def _fisher_claude_vs_others_contested_claim(
    judgments_by_model: dict[str, list[dict]],
    claude_model_key: str = "claude_sonnet46",
) -> list[dict]:
    if claude_model_key not in judgments_by_model:
        return []
    records = []
    claude_j = judgments_by_model[claude_model_key]
    n_c, s_c, _ = sycophancy_breakdown(claude_j, domain_filter="contested_claim")
    f_c = n_c - s_c
    for mk, judgments in judgments_by_model.items():
        if mk == claude_model_key:
            continue
        n_o, s_o, _ = sycophancy_breakdown(judgments, domain_filter="contested_claim")
        f_o = n_o - s_o
        if n_c == 0 or n_o == 0:
            rec = {
                "model_a": claude_model_key,
                "model_b": mk,
                "domain": "contested_claim",
                "n_a": n_c,
                "n_b": n_o,
                "success_a": s_c,
                "success_b": s_o,
                "insufficient_data": True,
                "p_value_raw": None,
            }
        else:
            rec = {
                "model_a": claude_model_key,
                "model_b": mk,
                "domain": "contested_claim",
                "n_a": n_c,
                "n_b": n_o,
                "success_a": s_c,
                "success_b": s_o,
                "rate_a": s_c / n_c,
                "rate_b": s_o / n_o,
                "insufficient_data": False,
                "p_value_raw": fisher_exact_2x2(s_c, f_c, s_o, f_o),
            }
        records.append(rec)
    adjusted = bonferroni_correction([r["p_value_raw"] for r in records])
    for r, p_adj in zip(records, adjusted):
        r["p_value_bonferroni"] = p_adj
    return records


def compute_significance_summary(
    model_keys: list[str],
    baseline_model: str | None = None,
    alpha: float = 0.05,
    ci_level: float = 0.95,
    n_bootstrap: int = 2000,
    n_permutations: int = 2000,
    min_n_binary: int = 5,
    min_n_continuous: int = 8,
    seed: int = 7,
) -> dict:
    if not model_keys:
        raise ValueError("model_keys must be non-empty")
    if baseline_model is None:
        baseline_model = model_keys[0]
    if baseline_model not in model_keys:
        raise ValueError(f"baseline_model {baseline_model!r} is not present in model_keys")

    rng = np.random.default_rng(seed)
    judgments_by_model = {mk: load_judgments(mk) for mk in model_keys}
    features_by_model = {mk: load_features(mk) for mk in model_keys}
    filters = _scenario_filters()

    all_metric_records = []
    main_records = []
    pairwise_records = []

    for scope_key, scope_label, where in filters:
        data_by_model = {
            mk: _collect_metric_data(judgments_by_model[mk], features_by_model[mk], where)
            for mk in model_keys
        }
        for mk in model_keys:
            if mk == baseline_model:
                continue
            for metric in METRIC_KEYS:
                rec = _compare_metric(
                    rng=rng,
                    metric_key=metric,
                    a_data=data_by_model[baseline_model][metric],
                    b_data=data_by_model[mk][metric],
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    ci_level=ci_level,
                    min_n=min_n_binary if metric == "sycophancy_rate" else min_n_continuous,
                )
                rec["comparison_family"] = "baseline_vs_others"
                rec["scope_key"] = scope_key
                rec["scope_label"] = scope_label
                rec["model_a"] = baseline_model
                rec["model_b"] = mk
                main_records.append(rec)
                all_metric_records.append(rec)

        for mka, mkb in itertools.combinations(model_keys, 2):
            for metric in METRIC_KEYS:
                rec = _compare_metric(
                    rng=rng,
                    metric_key=metric,
                    a_data=data_by_model[mka][metric],
                    b_data=data_by_model[mkb][metric],
                    n_bootstrap=n_bootstrap,
                    n_permutations=n_permutations,
                    ci_level=ci_level,
                    min_n=min_n_binary if metric == "sycophancy_rate" else min_n_continuous,
                )
                rec["comparison_family"] = "pairwise"
                rec["scope_key"] = scope_key
                rec["scope_label"] = scope_label
                rec["model_a"] = mka
                rec["model_b"] = mkb
                pairwise_records.append(rec)
                all_metric_records.append(rec)

    _apply_fdr(main_records)
    _apply_fdr(pairwise_records)
    _apply_fdr(all_metric_records)
    fisher_domain_pairwise = _fisher_domain_pairwise_per_model(judgments_by_model)
    fisher_claude_vs_others = _fisher_claude_vs_others_contested_claim(judgments_by_model)

    return {
        "metadata": {
            "models": model_keys,
            "baseline_model": baseline_model,
            "alpha": alpha,
            "ci_level": ci_level,
            "n_bootstrap": n_bootstrap,
            "n_permutations": n_permutations,
            "min_n_binary": min_n_binary,
            "min_n_continuous": min_n_continuous,
            "seed": seed,
            "reporting_policy": {
                "primary": "effect_sizes_and_confidence_intervals",
                "secondary": "fdr_corrected_p_values",
            },
        },
        "results": {
            "main_baseline_vs_others": main_records,
            "appendix_pairwise": pairwise_records,
            "all_tests_combined": all_metric_records,
            "fisher_domain_pairwise_per_model": fisher_domain_pairwise,
            "fisher_claude_vs_others_contested_claim": fisher_claude_vs_others,
        },
    }
