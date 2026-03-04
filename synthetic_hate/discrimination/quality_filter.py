"""
Step 5 — Quality Filtering: threshold-based filtering + manifest builder.

Usage:
    python discrimination/quality_filter.py

Inputs:
    data/synthetic_scored.csv
    data/real_samples.csv

Outputs:
    data/filtered/synthetic_top{N}.csv   (N = 90, 70, 50, 30, 10)
    data/filtered/synthetic_unfiltered.csv
    data/filtered/real_plus_top50.csv
    data/filtered/manifest.json
"""

import os
import sys
import json
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FILTERED_DIR = os.path.join(DATA_DIR, "filtered")
os.makedirs(FILTERED_DIR, exist_ok=True)

THRESHOLDS = [0.9, 0.7, 0.5, 0.3, 0.1]


def top_pct_filter(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """
    For each class separately, keep the top pct% of rows by quality_score.
    This preserves class balance and avoids one class being filtered more.
    """
    kept_parts = []
    for label in [1, 0]:
        subset = df[df["label"] == label].copy()
        n_keep = max(1, int(len(subset) * pct))
        kept = subset.nlargest(n_keep, "quality_score")
        kept_parts.append(kept)
    return pd.concat(kept_parts, ignore_index=True)


def main():
    scored_path = os.path.join(DATA_DIR, "synthetic_scored.csv")
    real_path = os.path.join(DATA_DIR, "real_samples.csv")

    if not os.path.exists(scored_path):
        print(f"ERROR: {scored_path} not found. Run discriminator_trainer.py first.")
        sys.exit(1)
    if not os.path.exists(real_path):
        print(f"ERROR: {real_path} not found. Run step1_data.py first.")
        sys.exit(1)

    synthetic_df = pd.read_csv(scored_path)
    real_df = pd.read_csv(real_path)

    print(f"Loaded {len(synthetic_df)} scored synthetic samples.")
    print(f"Loaded {len(real_df)} real samples.")

    manifest_variants = []

    # Add real_only to manifest (no file written — just reference)
    manifest_variants.append({
        "name": "real_only",
        "path": os.path.join(DATA_DIR, "real_samples.csv").replace("\\", "/"),
        "n_samples": len(real_df),
        "synthetic_pct": 0,
        "threshold": None,
    })

    # Save unfiltered
    unfiltered_path = os.path.join(FILTERED_DIR, "synthetic_unfiltered.csv")
    synthetic_df.to_csv(unfiltered_path, index=False)
    print(f"  synthetic_unfiltered -> {len(synthetic_df)} samples")
    manifest_variants.append({
        "name": "synthetic_unfiltered",
        "path": unfiltered_path.replace("\\", "/"),
        "n_samples": len(synthetic_df),
        "threshold": None,
        "synthetic_pct": 1.0,
    })

    # Threshold-based filtering
    top50_df = None
    for t in THRESHOLDS:
        pct_label = int(t * 100)
        filtered = top_pct_filter(synthetic_df, t)
        out_path = os.path.join(FILTERED_DIR, f"synthetic_top{pct_label}.csv")
        filtered.to_csv(out_path, index=False)
        print(f"  synthetic_top{pct_label} -> {len(filtered)} samples "
              f"(hate: {(filtered.label==1).sum()} | non-hate: {(filtered.label==0).sum()})")

        manifest_variants.append({
            "name": f"synthetic_top{pct_label}",
            "path": out_path.replace("\\", "/"),
            "n_samples": len(filtered),
            "threshold": t,
            "synthetic_pct": 1.0,
        })

        if t == 0.5:
            top50_df = filtered.copy()

    # Real + top-50% combined
    if top50_df is not None:
        combined_df = pd.concat([real_df, top50_df], ignore_index=True)
        combined_path = os.path.join(FILTERED_DIR, "real_plus_top50.csv")
        combined_df.to_csv(combined_path, index=False)
        n_real = len(real_df)
        n_syn = len(top50_df)
        print(f"  real_plus_top50 -> {len(combined_df)} samples "
              f"(real: {n_real} | synthetic: {n_syn})")
        manifest_variants.append({
            "name": "real_plus_top50",
            "path": combined_path.replace("\\", "/"),
            "n_samples": len(combined_df),
            "threshold": 0.5,
            "synthetic_pct": n_syn / len(combined_df),
        })

    # Save manifest
    manifest = {"variants": manifest_variants}
    manifest_path = os.path.join(FILTERED_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[OK] Manifest saved -> {manifest_path}")
    print(f"   {len(manifest_variants)} variants registered.")

    # Print manifest summary
    print("\n--- Manifest Summary ---")
    print(f"{'Name':<25} | {'N Samples':>10} | {'Threshold':>10} | {'Synthetic%':>10}")
    print("-" * 65)
    for v in manifest_variants:
        t_str = str(v.get("threshold", "—")) if v.get("threshold") is not None else "—"
        pct_str = f"{v.get('synthetic_pct', 0):.0%}"
        print(f"{v['name']:<25} | {v['n_samples']:>10} | {t_str:>10} | {pct_str:>10}")


if __name__ == "__main__":
    main()
