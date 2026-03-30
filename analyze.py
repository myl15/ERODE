# analyze.py
"""
Generate cross-model comparison plots for the report and presentation.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11

from config import ANALYSIS_DIR, JUDGMENTS_DIR, FEATURES_DIR


def plot_sycophancy_by_domain(summary, model_keys, outpath):
    """Bar chart: Sycophancy Rate by domain, grouped by model."""
    domains = ["Established Fact", "Contested Claim", "Subjective Judgment"]
    x = np.arange(len(domains))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, mk in enumerate(model_keys):
        rates = [summary[mk].get(d, {}).get("sycophancy_rate", 0) or 0 for d in domains]
        ax.bar(x + i * width, [r * 100 for r in rates], width, label=mk)
    
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_title("Sycophancy Rate by Knowledge Domain")
    ax.set_xticks(x + width)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sycophancy_by_pressure(summary, model_keys, outpath):
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
        ax.bar(x + i * width, [r * 100 for r in rates], width, label=mk)
    
    ax.set_ylabel("Sycophancy Rate (%)")
    ax.set_title("Sycophancy Rate by Pressure Type")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hedge_trajectory(features_by_model, model_keys, outpath):
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
        ax.plot(turns, means, marker="o", linewidth=2, markersize=8, label=mk)
    
    ax.set_ylabel("Mean Hedge Density (hedges per word)")
    ax.set_xlabel("Dialogue Turn")
    ax.set_title("Hedge Density Trajectory Under Pressure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pressure_response_curve(judgments_by_model, model_keys, outpath):
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
            ax.bar(x + i * width, rates, width, label=mk)
        
        ax.set_title(f"Position Distribution at {turn}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(positions, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% of Dialogues")
        ax.legend(fontsize=8)
    
    plt.suptitle("Pressure-Response Curve: Position State Distributions", fontsize=13)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ers_radar(summary, model_keys, outpath):
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
        ax.plot(angles, values, linewidth=2, label=mk)
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


def load_judgments_list(model_key):
    path = os.path.join(JUDGMENTS_DIR, f"{model_key}_judgments.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]

def load_features_list(model_key):
    path = os.path.join(FEATURES_DIR, f"{model_key}_features.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f]