"""
Step 7 — Figure Generator: 4 publication-ready figures.

Usage:
    python evaluation/figure_generator.py

Inputs:
    results/results.json
    data/synthetic_scored.csv  (for quality_dist figure)

Outputs (PDF + PNG at 300dpi in evaluation/figures/):
    bar_chart.pdf / .png          — F1-macro per variant, horizontal bar chart
    pareto_curve.pdf / .png       — Quality-Quantity Pareto Frontier (KEY FIGURE)
    quality_dist.pdf / .png       — Quality score overlapping histograms
    ablation.pdf / .png           — real_only vs real_plus_top50 grouped bar chart

All figures: figsize=(3.5, 2.8), seaborn colorblind palette, 10pt axis labels, 11pt titles.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Style Config ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=0.9)
CB_PALETTE = sns.color_palette("colorblind")
FIGSIZE = (3.5, 2.8)
DPI = 300


def save_fig(fig, name: str):
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  [OK] Saved {path}")
    plt.close(fig)


def load_results() -> pd.DataFrame:
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run experiment_runner.py first.")
        sys.exit(1)
    with open(results_path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ─── Figure 1 ────────────────────────────────────────────────────────────────

def figure1_bar_chart(df: pd.DataFrame):
    """Horizontal bar chart of F1-macro per variant, sorted descending."""
    df_sorted = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)

    # Assign bar colors
    def bar_color(name: str):
        if name == "real_only":
            return CB_PALETTE[2]          # green
        elif name == "real_plus_top50":
            return CB_PALETTE[1]          # orange
        else:
            return CB_PALETTE[4]          # purple

    colors = [bar_color(n) for n in df_sorted["variant_name"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.barh(
        df_sorted["variant_name"],
        df_sorted["f1_macro"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Real-only baseline dashed line
    real_baseline = df[df["variant_name"] == "real_only"]["f1_macro"].values
    if len(real_baseline) > 0:
        ax.axvline(
            real_baseline[0], color="black", linestyle="--", linewidth=1, label=f"Real-only ({real_baseline[0]:.2f})"
        )

    # Label each bar
    for bar, val in zip(bars, df_sorted["f1_macro"]):
        ax.text(
            val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=7.5
        )

    ax.set_xlabel("F1-macro", fontsize=10)
    ax.set_title("F1-macro by Training Variant", fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(0, df_sorted["f1_macro"].max() + 0.05)
    fig.tight_layout()
    save_fig(fig, "bar_chart")


# ─── Figure 2 ────────────────────────────────────────────────────────────────

def figure2_pareto_curve(df: pd.DataFrame):
    """Quality-Quantity Pareto Frontier (key paper figure)."""
    # Pareto variants: those trained on synthetic only with a defined threshold
    synth_variants = df[df["variant_name"].str.startswith("synthetic_top")].copy()
    unfiltered = df[df["variant_name"] == "synthetic_unfiltered"].copy()
    real_only = df[df["variant_name"] == "real_only"].copy()

    # Map variant to pct retained
    pct_map = {
        "synthetic_top10": 10,
        "synthetic_top30": 30,
        "synthetic_top50": 50,
        "synthetic_top70": 70,
        "synthetic_top90": 90,
    }

    synth_variants["pct_retained"] = synth_variants["variant_name"].map(pct_map)
    if len(unfiltered) > 0:
        unfiltered_row = unfiltered.iloc[[0]].copy()
        unfiltered_row["pct_retained"] = 100
        synth_variants = pd.concat([synth_variants, unfiltered_row], ignore_index=True)

    synth_variants = synth_variants.dropna(subset=["pct_retained"]).sort_values("pct_retained")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        synth_variants["pct_retained"], synth_variants["f1_macro"],
        marker="o", linewidth=1.8, markersize=5, color=CB_PALETTE[0], label="Filtered synthetic"
    )

    # Real-only baseline
    if len(real_only) > 0:
        baseline = real_only["f1_macro"].values[0]
        ax.axhline(baseline, color=CB_PALETTE[2], linestyle="--", linewidth=1.2, label=f"Real-only ({baseline:.3f})")

    # Annotate peak
    if len(synth_variants) > 0:
        peak_row = synth_variants.loc[synth_variants["f1_macro"].idxmax()]
        ax.annotate(
            f"Peak\n({int(peak_row['pct_retained'])}%, {peak_row['f1_macro']:.3f})",
            xy=(peak_row["pct_retained"], peak_row["f1_macro"]),
            xytext=(peak_row["pct_retained"] - 20, peak_row["f1_macro"] - 0.035),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=7,
        )

    ax.set_xlabel("% Synthetic Data Retained", fontsize=10)
    ax.set_ylabel("F1-macro", fontsize=10)
    ax.set_title("Quality-Quantity Pareto Frontier", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, 105)
    fig.tight_layout()
    save_fig(fig, "pareto_curve")


# ─── Figure 3 ────────────────────────────────────────────────────────────────

def figure3_quality_dist():
    """Overlapping histograms of quality scores with filtering threshold lines."""
    scored_path = os.path.join(DATA_DIR, "synthetic_scored.csv")
    if not os.path.exists(scored_path):
        print(f"[WARN]  Skipping quality_dist: {scored_path} not found.")
        return

    df = pd.read_csv(scored_path)
    palette = sns.color_palette("colorblind", 2)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label_val, label_name, color in [(1, "hate", palette[0]), (0, "non-hate", palette[1])]:
        subset = df[df["label"] == label_val]["quality_score"]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label_name, density=True)

    # Vertical dashed lines at thresholds
    for t in thresholds:
        # Get actual quality score percentile corresponding to top-t% per class
        # (quantile for threshold lines: keep top t%, so threshold = (1-t) quantile)
        hate_q = df[df["label"] == 1]["quality_score"].quantile(1 - t)
        ax.axvline(hate_q, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(hate_q + 0.005, ax.get_ylim()[1] * 0.85, f"{int(t*100)}%", fontsize=6, color="grey")

    ax.set_xlabel("Quality Score P(real | text)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Quality Score Distribution (Synthetic)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "quality_dist")


# ─── Figure 4 ────────────────────────────────────────────────────────────────

def figure4_ablation(df: pd.DataFrame):
    """Grouped bar chart: real_only vs real_plus_top50 across 4 metrics."""
    metrics = ["f1_macro", "precision", "recall", "accuracy"]
    metric_labels = ["F1-macro", "Precision", "Recall", "Accuracy"]

    real_row = df[df["variant_name"] == "real_only"]
    comb_row = df[df["variant_name"] == "real_plus_top50"]

    if real_row.empty or comb_row.empty:
        print("[WARN]  Skipping ablation: 'real_only' or 'real_plus_top50' not in results.")
        return

    real_vals = [real_row[m].values[0] for m in metrics]
    comb_vals = [comb_row[m].values[0] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars1 = ax.bar(x - width / 2, real_vals, width, label="Real only", color=CB_PALETTE[2], edgecolor="white")
    bars2 = ax.bar(x + width / 2, comb_vals, width, label="Real + top-50%", color=CB_PALETTE[1], edgecolor="white")

    # Annotate % improvement above each pair
    for i, (rv, cv) in enumerate(zip(real_vals, comb_vals)):
        if rv > 0:
            pct_imp = (cv - rv) / rv * 100
            sign_str = f"+{pct_imp:.1f}%" if pct_imp >= 0 else f"{pct_imp:.1f}%"
            max_y = max(rv, cv)
            ax.text(i, max_y + 0.01, sign_str, ha="center", fontsize=6.5, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Real only vs. Real + Filtered Synthetic", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    save_fig(fig, "ablation")


def main():
    print("Loading results...")
    df = load_results()
    print(f"  Loaded {len(df)} experiment results.")

    print("\n--- Figure 1: Bar Chart ---")
    figure1_bar_chart(df)

    print("\n--- Figure 2: Pareto Curve ---")
    figure2_pareto_curve(df)

    print("\n--- Figure 3: Quality Distribution ---")
    figure3_quality_dist()

    print("\n--- Figure 4: Ablation ---")
    figure4_ablation(df)

    print(f"\n[OK] All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
