"""
Step 4 — Quality Discriminator: real vs. synthetic classifier.

Usage:
    python discrimination/discriminator_trainer.py

Inputs:
    data/real_samples.csv
    data/synthetic_clean.csv

Outputs:
    discrimination/discriminator.pkl    — best sklearn model
    data/synthetic_scored.csv           — synthetic_clean + quality_score column
    evaluation/figures/quality_dist.png — quality score histogram by hate/non-hate

Core framing:
    Train a binary classifier: real (1) vs. synthetic (0).
    quality_score = P(real | text) — higher means more human-like.
    Samples indistinguishable from real data (score ≈ 0.5) are highest quality.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DISC_DIR = os.path.join(BASE_DIR, "discrimination")
FIGURES_DIR = os.path.join(BASE_DIR, "evaluation", "figures")
SEED = 42

os.makedirs(DISC_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def encode(model, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def cv_eval(model_obj, X_train, y_train, name: str) -> dict:
    """5-fold stratified CV -> AUC-ROC + F1-macro."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        model_obj, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1_macro"],
        n_jobs=-1,
    )
    auc = scores["test_roc_auc"].mean()
    f1 = scores["test_f1_macro"].mean()
    print(f"  {name:<30} 5-fold AUC: {auc:.4f} | F1-macro: {f1:.4f}")
    return {"name": name, "model": model_obj, "auc": auc, "f1": f1}


def plot_quality_distribution(synthetic_scored: pd.DataFrame):
    """Overlapping histograms of quality_score for hate vs. non-hate."""
    palette = sns.color_palette("colorblind", 2)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label_val, label_name, color in [
        (1, "hate", palette[0]),
        (0, "non-hate", palette[1]),
    ]:
        subset = synthetic_scored[synthetic_scored["label"] == label_val]["quality_score"]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label_name, density=True)

    ax.set_xlabel("Quality Score P(real | text)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Quality Score Distribution (Synthetic Data)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "quality_dist.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Quality distribution plot saved -> {save_path}")


def main():
    real_path = os.path.join(DATA_DIR, "real_samples.csv")
    synth_path = os.path.join(DATA_DIR, "synthetic_clean.csv")

    if not os.path.exists(real_path):
        print(f"ERROR: {real_path} not found. Run step1_data.py first.")
        sys.exit(1)
    if not os.path.exists(synth_path):
        print(f"ERROR: {synth_path} not found. Run diversity_checker.py first.")
        sys.exit(1)

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    print(f"Real samples: {len(real_df)}")
    print(f"Synthetic samples: {len(synth_df)}")

    # Discriminator labels: real=1, synthetic=0
    real_df["disc_label"] = 1
    synth_df["disc_label"] = 0

    combined = pd.concat(
        [real_df[["text", "disc_label"]], synth_df[["text", "disc_label"]]],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nCombined dataset: {len(combined)} samples (real=1, synthetic=0)")

    # Encode
    model = load_sentence_model()
    all_texts = combined["text"].tolist()
    print("\nEncoding all texts...")
    X = encode(model, all_texts)
    y = combined["disc_label"].values

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Define candidates
    candidates = [
        ("LogisticRegression", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ("RandomForestClassifier", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
    ]

    # 5-fold CV on train
    print("\n--- 5-fold Stratified CV on Training Set ---")
    results = []
    for name, clf in candidates:
        r = cv_eval(clf, X_train, y_train, name)
        results.append(r)

    # Select best by AUC
    best = max(results, key=lambda r: r["auc"])
    best_name = best["name"]
    best_clf = best["model"]

    print(f"\n[BEST] Best model: {best_name} | AUC: {best['auc']:.4f} | F1: {best['f1']:.4f}")

    # Retrain on full train set
    print(f"Retraining {best_name} on full train set...")
    best_clf.fit(X_train, y_train)

    # Evaluate on held-out test
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Test AUC: {test_auc:.4f} | Test F1-macro: {test_f1:.4f}")

    # Save discriminator
    disc_path = os.path.join(DISC_DIR, "discriminator.pkl")
    joblib.dump({"model": best_clf, "model_name": best_name}, disc_path)
    print(f"[OK] Discriminator saved -> {disc_path}")

    # Score ALL synthetic samples
    print("\nScoring all synthetic samples...")
    synth_texts = synth_df["text"].tolist()
    synth_embeddings = encode(model, synth_texts)
    synth_df["quality_score"] = best_clf.predict_proba(synth_embeddings)[:, 1]

    scored_path = os.path.join(DATA_DIR, "synthetic_scored.csv")
    synth_df.to_csv(scored_path, index=False)
    print(f"[OK] Scored data saved -> {scored_path}")

    # Print summary stats
    print(f"\nBest model: {best_name} | AUC: {best['auc']:.2f} | F1: {best['f1']:.2f}")
    print("Quality score stats (synthetic):")
    for label_val, label_name in [(1, "hate"), (0, "non-hate")]:
        subset = synth_df[synth_df["label"] == label_val]["quality_score"]
        print(
            f"  {label_name:<10} -> mean: {subset.mean():.2f} | std: {subset.std():.2f} | "
            f"min: {subset.min():.2f} | max: {subset.max():.2f}"
        )

    # Plot quality distribution
    plot_quality_distribution(synth_df)


if __name__ == "__main__":
    main()
