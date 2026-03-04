"""
Step 7 — Figure Generator: 4 publication-ready figures.

Outputs (PDF + PNG):
  evaluation/figures/pareto_curve.{pdf,png}
  evaluation/figures/pr_asymmetry.{pdf,png}
  evaluation/figures/performance_comparison.{pdf,png}
  evaluation/figures/quality_score_dist.{pdf,png}
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

RESULTS = [
    {"variant_name": "real_only",            "threshold": None, "synthetic_pct": 0,    "f1_macro": 0.6511, "precision": 0.6783, "recall": 0.66},
    {"variant_name": "synthetic_unfiltered", "threshold": None, "synthetic_pct": 1.0,  "f1_macro": 0.5703, "precision": 0.7232, "recall": 0.62},
    {"variant_name": "synthetic_top90",      "threshold": 0.9,  "synthetic_pct": 1.0,  "f1_macro": 0.5091, "precision": 0.6894, "recall": 0.58},
    {"variant_name": "synthetic_top70",      "threshold": 0.7,  "synthetic_pct": 1.0,  "f1_macro": 0.5328, "precision": 0.6765, "recall": 0.59},
    {"variant_name": "synthetic_top50",      "threshold": 0.5,  "synthetic_pct": 1.0,  "f1_macro": 0.5404, "precision": 0.7076, "recall": 0.60},
    {"variant_name": "synthetic_top30",      "threshold": 0.3,  "synthetic_pct": 1.0,  "f1_macro": 0.5413, "precision": 0.5933, "recall": 0.57},
    {"variant_name": "synthetic_top10",      "threshold": 0.1,  "synthetic_pct": 1.0,  "f1_macro": 0.3333, "precision": 0.25,   "recall": 0.50},
    {"variant_name": "real_plus_top50",      "threshold": 0.5,  "synthetic_pct": 0.54, "f1_macro": 0.6576, "precision": 0.6987, "recall": 0.67},
]

# Set global style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
PALETTE = sns.color_palette("colorblind")
REAL_BASELINE = 0.6511


def savefig(fig, name):
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[OK] {name}.{{pdf,png}} saved -> {FIGURES_DIR}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Pareto Curve
# ─────────────────────────────────────────────────────────────────────────────
def figure1_pareto():
    # Only pure-synthetic conditions for the sweep
    sweep = [
        (10,  0.3333),
        (30,  0.5413),
        (50,  0.5404),
        (70,  0.5328),
        (90,  0.5091),
        (100, 0.5703),   # unfiltered = 100% retained
    ]
    x_vals = [s[0] for s in sweep]
    y_vals = [s[1] for s in sweep]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Shaded below-baseline region
    ax.axhline(REAL_BASELINE, color="#e74c3c", linestyle="--", linewidth=1.4, zorder=3)
    ax.fill_between([0, 105], [REAL_BASELINE, REAL_BASELINE], [0.25, 0.25],
                    color="#e74c3c", alpha=0.06, label="_nolegend_")

    # Pareto line
    ax.plot(x_vals, y_vals, "o-", color=PALETTE[0], linewidth=2,
            markersize=6, markerfacecolor="white", markeredgewidth=1.8, zorder=4)

    # Baseline label
    ax.text(102, REAL_BASELINE + 0.005, "Real Data\nBaseline", fontsize=7,
            color="#e74c3c", va="bottom", ha="left")

    # Annotation
    ax.annotate("No significant\nimprovement\n(ANOVA p>0.05)",
                xy=(50, 0.5404), xytext=(18, 0.43),
                fontsize=6.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    ax.set_xlim(5, 110)
    ax.set_ylim(0.25, 0.72)
    ax.set_xlabel("% Synthetic Data Retained by Quality Filter", fontsize=8)
    ax.set_ylabel("F1-macro", fontsize=8)
    ax.set_title("Pareto Analysis: Quality Filtering vs.\nClassification Performance", fontsize=8.5, pad=4)
    ax.set_xticks([10, 30, 50, 70, 90, 100])
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    savefig(fig, "pareto_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Precision-Recall Asymmetry
# ─────────────────────────────────────────────────────────────────────────────
def figure2_pr_asymmetry():
    # Order ascending by synthetic_pct, real_only last
    ordered = [
        ("top10",      0.3333, 0.25,   0.50),
        ("top30",      0.5413, 0.5933, 0.57),
        ("top50",      0.5404, 0.7076, 0.60),
        ("top70",      0.5328, 0.6765, 0.59),
        ("top90",      0.5091, 0.6894, 0.58),
        ("unfiltered", 0.5703, 0.7232, 0.62),
        ("aug+50",     0.6576, 0.6987, 0.67),
        ("real only",  0.6511, 0.6783, 0.66),
    ]
    labels   = [o[0] for o in ordered]
    precs    = [o[2] for o in ordered]
    recs     = [o[3] for o in ordered]
    x        = np.arange(len(labels))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars_p = ax.bar(x - width/2, precs, width, label="Precision", color=PALETTE[0], alpha=0.85)
    bars_r = ax.bar(x + width/2, recs,  width, label="Recall",    color=PALETTE[1], alpha=0.85)

    # Annotate PR-gap brackets for synthetic conditions
    for i in range(6):   # first 6 are synthetic
        p, r = precs[i], recs[i]
        if p > r:
            ax.annotate("", xy=(x[i]+width/2, r), xytext=(x[i]-width/2, p),
                        arrowprops=dict(arrowstyle="-", color="#999999",
                                        connectionstyle="arc3,rad=0", lw=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.5)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score", fontsize=8)
    ax.set_title("Precision-Recall Asymmetry\nAcross Training Conditions", fontsize=8.5, pad=4)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    ax.yaxis.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "pr_asymmetry")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Performance Comparison (horizontal bar chart)
# ─────────────────────────────────────────────────────────────────────────────
def figure3_performance():
    df = pd.DataFrame(RESULTS)
    df = df.sort_values("f1_macro", ascending=True).reset_index(drop=True)

    def bar_color(row):
        if row["variant_name"] == "real_only":
            return PALETTE[2]           # green
        if row["variant_name"] == "real_plus_top50":
            return PALETTE[3]           # orange/combined
        return PALETTE[0]               # blue/purple for pure synthetic

    colors = [bar_color(r) for _, r in df.iterrows()]
    labels = [n.replace("synthetic_", "synth_").replace("_", " ") for n in df["variant_name"]]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    bars = ax.barh(labels, df["f1_macro"], color=colors, edgecolor="white", linewidth=0.5)

    # Baseline dashed line
    ax.axvline(REAL_BASELINE, color="#e74c3c", linestyle="--", linewidth=1.2, zorder=3)
    ax.text(REAL_BASELINE + 0.003, -0.6, "Baseline", fontsize=6.5, color="#e74c3c", va="bottom")

    # Value labels
    for bar, val in zip(bars, df["f1_macro"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=6.5)

    ax.set_xlim(0.0, 0.74)
    ax.set_xlabel("F1-macro", fontsize=8)
    ax.set_title("Performance Comparison\nAcross Training Conditions", fontsize=8.5, pad=4)
    ax.tick_params(labelsize=7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[2], label="Real data only"),
        Patch(facecolor=PALETTE[3], label="Real + synthetic"),
        Patch(facecolor=PALETTE[0], label="Synthetic only"),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc="lower right")
    ax.xaxis.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(fig, "performance_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Quality Score Distribution
# ─────────────────────────────────────────────────────────────────────────────
def figure4_quality_dist():
    scored_path = os.path.join(DATA_DIR, "synthetic_scored.csv")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    if os.path.exists(scored_path):
        df   = pd.read_csv(scored_path)
        hate     = df[df["label"] == 1]["quality_score"].dropna()
        non_hate = df[df["label"] == 0]["quality_score"].dropna()
        ax.hist(hate,     bins=30, alpha=0.6, color=PALETTE[1], density=True, label="Hate")
        ax.hist(non_hate, bins=30, alpha=0.6, color=PALETTE[0], density=True, label="Non-hate")
        ax.legend(fontsize=7)
    else:
        # Fallback: simulate from known discriminator stats
        # hate:     mean=0.04, std=0.05
        # non-hate: mean=0.03, std=0.03
        rng      = np.random.default_rng(42)
        hate     = rng.normal(0.04, 0.05, 1000).clip(0, 1)
        non_hate = rng.normal(0.03, 0.03, 1000).clip(0, 1)
        ax.hist(hate,     bins=30, alpha=0.6, color=PALETTE[1], density=True, label="Hate (simulated)")
        ax.hist(non_hate, bins=30, alpha=0.6, color=PALETTE[0], density=True, label="Non-hate (simulated)")
        ax.legend(fontsize=7)

    # Threshold lines
    for thresh, ls in [(0.1, ":"), (0.3, "-."), (0.5, "--"), (0.7, "-."), (0.9, ":")]:
        ax.axvline(thresh, color="#666666", linestyle=ls, linewidth=0.8, alpha=0.7)
        ax.text(thresh, ax.get_ylim()[1] * 0.88, f"{thresh}", fontsize=5.5,
                ha="center", color="#555555")

    ax.set_xlabel("Quality Score P(real | text)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("Discriminator Quality Score Distribution\n"
                 "(Filtering Thresholds Shown as Dashed Lines)", fontsize=8.5, pad=4)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    savefig(fig, "quality_score_dist")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating 4 publication figures...")
    figure1_pareto()
    figure2_pr_asymmetry()
    figure3_performance()
    figure4_quality_dist()
    print("\n[DONE] All figures generated.")
