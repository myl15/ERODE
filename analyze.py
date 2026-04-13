# analyze.py
"""
Generate cross-model comparison plots for the report and presentation.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patheffects as pe

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

from config import ANALYSIS_DIR, JUDGMENTS_DIR, FEATURES_DIR
from metrics import DOMAINS, PRESSURE_TYPES, RUBRIC_DIMS

POSITION_ORDER = ["MAINTAIN", "SOFTEN", "EQUIVOCATE", "CONCEDE", "CAPITULATE"]
DOMAIN_AXIS_LABELS = ["Established\nFact", "Contested\nClaim", "Subjective\nJudgment"]
PRESSURE_AXIS_LABELS = ["Polite\ndisagree.", "Confident\ncontradict.", "Appeal to\nauthority", "Emotional/\nsocial"]
RUBRIC_SHORT_LABELS = {
    "epistemic_consistency": "Epistemic\nconsistency",
    "evidential_reasoning": "Evidential\nreasoning",
    "pressure_resistance": "Pressure\nresistance",
    "appropriate_flexibility": "Appropriate\nflexibility",
}


def _legend_label(model_key, model_labels):
    if model_labels is None:
        return model_key
    return model_labels.get(model_key, model_key)


def plot_sycophancy_by_domain(summary, model_keys, outpath, model_labels=None):
    """Bar chart: Sycophancy Rate by domain, grouped by model."""
    domains = ["Established Fact", "Contested Claim", "Subjective Judgment"]
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, mk in enumerate(model_keys):
        rates = [summary[mk].get(d, {}).get("sycophancy_rate", 0) or 0 for d in domains]
        ax.bar(x + i * width, [r * 100 for r in rates], width, label=_legend_label(mk, model_labels))
    
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_title("Sycophancy Rate by Knowledge Domain")
    ax.set_xticks(x + width)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sycophancy_by_pressure(summary, model_keys, outpath, model_labels=None):
    """Bar chart: Sycophancy Rate by pressure type, grouped by model."""
    pressure_types = [
        "polite_disagreement", "confident_contradiction",
        "appeal_to_authority", "emotional_social",
    ]
    labels = ["Polite\nDisagreement", "Confident\nContradiction",
              "Appeal to\nAuthority", "Emotional/\nSocial"]
    x = np.arange(len(pressure_types))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, mk in enumerate(model_keys):
        rates = [summary[mk].get(pt, {}).get("sycophancy_rate", 0) or 0 for pt in pressure_types]
        ax.bar(x + i * width, [r * 100 for r in rates], width, label=_legend_label(mk, model_labels))
    
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_title("Sycophancy Rate by Pressure Type")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hedge_trajectory(features_by_model, model_keys, outpath, model_labels=None):
    """
    Line plot: Mean hedge density across turns (T1, T3, T5) per model.
    This is the "capitulation fingerprint" — your most novel visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    turns = ["T1", "T3", "T5"]
    
    for mk in model_keys:
        means = []
        for t in turns:
            densities = [f[t]["hedge_density"] for f in features_by_model[mk] if t in f]
            means.append(np.mean(densities) if densities else 0)
        ax.plot(
            turns,
            means,
            marker="o",
            linewidth=2,
            markersize=8,
            label=_legend_label(mk, model_labels),
        )
    
    ax.set_ylabel("Mean Hedge Density (hedges per word)")
    ax.set_xlabel("Dialogue Turn")
    ax.set_title("Hedge Density Trajectory Under Pressure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pressure_response_curve(judgments_by_model, model_keys, outpath, model_labels=None):
    """
    Heatmap or grouped bar: Position state distribution at T3 and T5 per model.
    Shows whether models degrade gradually or catastrophically.
    """
    positions = ["MAINTAIN", "SOFTEN", "EQUIVOCATE", "CONCEDE", "CAPITULATE"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    for idx, turn in enumerate(["T3", "T5"]):
        ax = axes[idx]
        x = np.arange(len(positions))
        width = 0.25
        
        for i, mk in enumerate(model_keys):
            counts = {p: 0 for p in positions}
            total = 0
            for j in judgments_by_model[mk]:
                pos = j.get("position_classification", {})
                if "error" not in pos and turn in pos:
                    p = pos[turn].get("position", "")
                    if p in counts:
                        counts[p] += 1
                        total += 1
            
            rates = [(counts[p] / total * 100) if total > 0 else 0 for p in positions]
            ax.bar(x + i * width, rates, width, label=_legend_label(mk, model_labels))
        
        ax.set_title(f"Position Distribution at {turn}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(positions, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% of Dialogues")
        ax.legend(fontsize=8)
    
    plt.suptitle("Pressure-Response Curve: Position State Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ers_radar(summary, model_keys, outpath, model_labels=None):
    """
    Radar chart showing the 4 rubric dimensions per model.
    Good for the presentation — visually striking and information-dense.
    """
    dimensions = ["epistemic_consistency", "evidential_reasoning",
                  "pressure_resistance", "appropriate_flexibility"]
    dim_labels = ["Epistemic\nConsistency", "Evidential\nReasoning",
                  "Pressure\nResistance", "Appropriate\nFlexibility"]
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    for mk in model_keys:
        judgments = load_judgments_list(mk)
        values = []
        for dim in dimensions:
            scores = [
                j["rubric_scores"][dim]["score"]
                for j in judgments
                if "error" not in j.get("rubric_scores", {}) and dim in j.get("rubric_scores", {})
            ]
            values.append(np.mean(scores) if scores else 0)
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=_legend_label(mk, model_labels))
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Epistemic Robustness Profile", pad=20)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_domain_pressure_heatmaps(summary, model_keys, outpath, model_labels=None):
    """Three heatmaps per model: SR (%), mean ERS, mean hedge drift (domain × pressure)."""
    n_models = len(model_keys)
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 3.8 * n_models), squeeze=False)

    for ri, mk in enumerate(model_keys):
        cells = summary[mk].get("domain_pressure_cells") or {}
        sr = np.full((len(DOMAINS), len(PRESSURE_TYPES)), np.nan)
        ers = np.full_like(sr, np.nan)
        hd = np.full_like(sr, np.nan)
        ann_sr = [[""] * len(PRESSURE_TYPES) for _ in DOMAINS]

        for i, d in enumerate(DOMAINS):
            row = cells.get(d) or {}
            for j, p in enumerate(PRESSURE_TYPES):
                c = row.get(p) or {}
                r = c.get("sycophancy_rate")
                if r is not None:
                    sr[i, j] = r * 100.0
                e = c.get("ERS")
                if e is not None:
                    ers[i, j] = e
                h = c.get("hedge_drift")
                if h is not None:
                    hd[i, j] = h
                ne = c.get("n_eligible", 0)
                ann_sr[i][j] = f"n={ne}"

        hedges_abs = float(np.nanmax(np.abs(hd)))
        if not np.isfinite(hedges_abs):
            hedges_abs = 1e-9
        else:
            hedges_abs = max(hedges_abs, 1e-9)

        panels = [
            (sr, "Sycophancy rate (%)", "viridis", 0.0, 100.0, True, ann_sr),
            (ers, "Mean ERS", "viridis", 1.0, 5.0, False, None),
            (hd, "Mean hedge drift (T5−T1 density)", "coolwarm", -hedges_abs, hedges_abs, False, None),
        ]

        for ci, (data, title, cmap, vmin, vmax, annotate, ann) in enumerate(panels):
            ax = axes[ri, ci]
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(PRESSURE_TYPES)))
            ax.set_yticks(np.arange(len(DOMAINS)))
            ax.set_xticklabels(PRESSURE_AXIS_LABELS, fontsize=8)
            ax.set_yticklabels(DOMAIN_AXIS_LABELS, fontsize=9)
            ax.set_title(title, fontsize=11)
            if annotate and ann:
                for i in range(len(DOMAINS)):
                    for j in range(len(PRESSURE_TYPES)):
                        if not np.isnan(data[i, j]):
                            t = f"{data[i, j]:.0f}%\n{ann[i][j]}"
                        else:
                            t = f"—\n{ann[i][j]}"
                        txt = ax.text(j, i, t, ha="center", va="center", color="white", fontsize=7)
                        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])
        axes[ri, 0].set_ylabel(_legend_label(mk, model_labels), fontsize=10, fontweight="bold")

    plt.suptitle("Domain × pressure interaction", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rubric_by_domain(judgments_by_model, model_keys, outpath, model_labels=None):
    """Grouped bars: mean rubric dimension by knowledge domain (one panel per model)."""
    n_dom = len(DOMAINS)
    n_dim = len(RUBRIC_DIMS)
    fig, axes = plt.subplots(1, len(model_keys), figsize=(5.5 * len(model_keys), 4.5), squeeze=False)

    for ax, mk in zip(axes[0], model_keys):
        x = np.arange(n_dom)
        width = 0.8 / n_dim
        for di, dim in enumerate(RUBRIC_DIMS):
            means = []
            for d in DOMAINS:
                scores = [
                    j["rubric_scores"][dim]["score"]
                    for j in judgments_by_model[mk]
                    if j.get("domain") == d
                    and "error" not in j.get("rubric_scores", {})
                    and dim in j.get("rubric_scores", {})
                ]
                means.append(np.mean(scores) if scores else 0.0)
            offset = (di - (n_dim - 1) / 2) * width
            ax.bar(x + offset, means, width, label=RUBRIC_SHORT_LABELS.get(dim, dim))

        ax.set_ylabel("Mean score (1–5)")
        ax.set_title(_legend_label(mk, model_labels))
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("_", " ").title() for d in DOMAINS])
        ax.set_ylim(0, 5.5)
        ax.legend(fontsize=8)

    fig.suptitle("Judge rubric means by domain", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rubric_by_pressure(judgments_by_model, model_keys, outpath, model_labels=None):
    """Grouped bars: mean rubric dimension by pressure type (one panel per model)."""
    n_pt = len(PRESSURE_TYPES)
    n_dim = len(RUBRIC_DIMS)
    fig, axes = plt.subplots(1, len(model_keys), figsize=(5.5 * len(model_keys), 4.5), squeeze=False)

    for ax, mk in zip(axes[0], model_keys):
        x = np.arange(n_pt)
        width = 0.8 / n_dim
        for di, dim in enumerate(RUBRIC_DIMS):
            means = []
            for p in PRESSURE_TYPES:
                scores = [
                    j["rubric_scores"][dim]["score"]
                    for j in judgments_by_model[mk]
                    if j.get("pressure_type") == p
                    and "error" not in j.get("rubric_scores", {})
                    and dim in j.get("rubric_scores", {})
                ]
                means.append(np.mean(scores) if scores else 0.0)
            offset = (di - (n_dim - 1) / 2) * width
            ax.bar(x + offset, means, width, label=RUBRIC_SHORT_LABELS.get(dim, dim))

        ax.set_ylabel("Mean score (1–5)")
        ax.set_title(_legend_label(mk, model_labels))
        ax.set_xticks(x)
        ax.set_xticklabels(PRESSURE_AXIS_LABELS)
        ax.set_ylim(0, 5.5)
        ax.legend(fontsize=8)

    fig.suptitle("Judge rubric means by pressure type", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def _transition_counts(judgments, t_from, t_to):
    mat = np.zeros((len(POSITION_ORDER), len(POSITION_ORDER)))
    for j in judgments:
        pos = j.get("position_classification", {})
        if "error" in pos:
            continue
        a = pos.get(t_from, {}).get("position", "")
        b = pos.get(t_to, {}).get("position", "")
        if a in POSITION_ORDER and b in POSITION_ORDER:
            mat[POSITION_ORDER.index(a), POSITION_ORDER.index(b)] += 1
    return mat


def plot_position_transition_matrices(judgments_by_model, model_keys, outpath, model_labels=None):
    """Row-normalized heatmaps for T1→T3 and T3→T5 transitions per model."""
    n = len(model_keys)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.2 * n), squeeze=False)

    for ri, mk in enumerate(model_keys):
        for ci, (ta, tb) in enumerate([("T1", "T3"), ("T3", "T5")]):
            ax = axes[ri, ci]
            mat = _transition_counts(judgments_by_model[mk], ta, tb)
            row_sums = mat.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                norm = np.where(row_sums > 0, mat / row_sums, 0.0)
            im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(np.arange(len(POSITION_ORDER)))
            ax.set_yticks(np.arange(len(POSITION_ORDER)))
            ax.set_xticklabels(POSITION_ORDER, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(POSITION_ORDER, fontsize=8)
            ax.set_xlabel(tb)
            ax.set_ylabel(ta)
            ax.set_title(f"{_legend_label(mk, model_labels)}: P({tb} | {ta}) row-normalized")
    plt.suptitle("Position transition matrices", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_t5_capitulation_given_t3(judgments_by_model, model_keys, outpath, model_labels=None):
    """P(T5 ∈ concede/capitulate | T3 = state) for each observed T3 state."""
    cap = {"CONCEDE", "CAPITULATE"}
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(POSITION_ORDER))
    width = 0.8 / max(len(model_keys), 1)

    for mi, mk in enumerate(model_keys):
        cond_p = {p: [] for p in POSITION_ORDER}
        for t3 in POSITION_ORDER:
            hit = 0
            tot = 0
            for j in judgments_by_model[mk]:
                pos = j.get("position_classification", {})
                if "error" in pos:
                    continue
                p3 = pos.get("T3", {}).get("position", "")
                p5 = pos.get("T5", {}).get("position", "")
                if p3 != t3 or not p5:
                    continue
                tot += 1
                if p5 in cap:
                    hit += 1
            cond_p[t3] = (100.0 * hit / tot) if tot else np.nan

        heights = [cond_p[p] for p in POSITION_ORDER]
        ax.bar(x + mi * width, heights, width, label=_legend_label(mk, model_labels))

    ax.set_ylabel("P(capitulate at T5 | T3 state) (%)")
    ax.set_title("Cascade: conditional sycophancy after mid-dialogue position")
    ax.set_xticks(x + width * (len(model_keys) - 1) / 2)
    ax.set_xticklabels(POSITION_ORDER, rotation=25, ha="right")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hedge_trajectory_by_domain(features_by_model, model_keys, outpath, model_labels=None):
    """Mean hedge density T1–T5, faceted by domain."""
    turns = ["T1", "T3", "T5"]
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(12, 4), sharey=True)
    for ax, dom in zip(axes, DOMAINS):
        for mk in model_keys:
            fs = [f for f in features_by_model[mk] if f.get("domain") == dom]
            means = []
            for t in turns:
                dens = [f[t]["hedge_density"] for f in fs if t in f and isinstance(f.get(t), dict)]
                means.append(float(np.mean(dens)) if dens else np.nan)
            ax.plot(turns, means, marker="o", linewidth=2, label=_legend_label(mk, model_labels))
        ax.set_title(dom.replace("_", " ").title())
        ax.set_xlabel("Turn")
    axes[0].set_ylabel("Mean hedge density")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(model_keys), bbox_to_anchor=(0.5, 1.08))
    plt.suptitle("Hedge trajectory by domain", y=1.05)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_lexical_trajectory_by_domain(features_by_model, model_keys, outpath, model_labels=None):
    """Booster density and mean word count by turn, faceted by domain."""
    turns = ["T1", "T3", "T5"]
    metrics = [
        ("booster_density", "Mean booster density"),
        ("word_count", "Mean word count"),
    ]
    fig, axes = plt.subplots(len(metrics), len(DOMAINS), figsize=(12, 6), sharex=True)
    for row, (key, ylabel) in enumerate(metrics):
        for col, dom in enumerate(DOMAINS):
            ax = axes[row, col]
            for mk in model_keys:
                fs = [f for f in features_by_model[mk] if f.get("domain") == dom]
                means = []
                for t in turns:
                    vals = [
                        f[t][key]
                        for f in fs
                        if t in f and isinstance(f.get(t), dict) and key in f[t]
                    ]
                    means.append(float(np.mean(vals)) if vals else np.nan)
                ax.plot(turns, means, marker="o", linewidth=2, label=_legend_label(mk, model_labels))
            if row == 0:
                ax.set_title(dom.replace("_", " ").title())
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == len(metrics) - 1:
                ax.set_xlabel("Turn")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(model_keys), bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("Booster density and response length by domain", y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def _norm_position_token(s):
    if not s:
        return ""
    u = s.strip().upper()
    return u if u in POSITION_ORDER else ""


def plot_heuristic_judge_confusion(
    features_by_model, judgments_by_model, model_keys, outpath, model_labels=None
):
    """Confusion matrix: judge T5 position (rows) vs lexical heuristic at T5 (cols)."""
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for idx, mk in enumerate(model_keys):
        ax = axes[0, idx]
        j_by = {j["scenario_id"]: j for j in judgments_by_model[mk]}
        mat = np.zeros((len(POSITION_ORDER), len(POSITION_ORDER)))
        for f in features_by_model[mk]:
            sid = f.get("scenario_id")
            j = j_by.get(sid)
            if not j:
                continue
            pos_j = j.get("position_classification", {})
            if "error" in pos_j:
                continue
            gold = _norm_position_token(pos_j.get("T5", {}).get("position", ""))
            t5f = f.get("T5")
            if not isinstance(t5f, dict):
                continue
            pred_raw = t5f.get("position_heuristic", "") or ""
            pred = _norm_position_token(pred_raw)
            if gold and pred:
                mat[POSITION_ORDER.index(gold), POSITION_ORDER.index(pred)] += 1

        row_sums = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.where(row_sums > 0, 100.0 * mat / row_sums, 0.0)
        im = ax.imshow(pct, aspect="auto", cmap="viridis", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(POSITION_ORDER)))
        ax.set_yticks(np.arange(len(POSITION_ORDER)))
        ax.set_xticklabels(POSITION_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(POSITION_ORDER, fontsize=8)
        ax.set_xlabel("Heuristic (T5)")
        ax.set_ylabel("Judge (T5)")
        for i in range(len(POSITION_ORDER)):
            for j in range(len(POSITION_ORDER)):
                c = int(mat[i, j])
                if c:
                    ax.text(j, i, f"{pct[i, j]:.0f}%\n({c})", ha="center", va="center", color="w", fontsize=7)
        ax.set_title(f"{_legend_label(mk, model_labels)}: row % of judge label")

    plt.suptitle("Lexical heuristic vs judge (T5)", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def load_judgments_list(model_key):
    path = os.path.join(JUDGMENTS_DIR, f"{model_key}_judgments.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_features_list(model_key):
    path = os.path.join(FEATURES_DIR, f"{model_key}_features.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]