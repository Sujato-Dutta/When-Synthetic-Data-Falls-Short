"""
Step 3 — Diversity Checker + Deduplication.

Usage:
    python generation/diversity_checker.py

Inputs:
    data/synthetic_raw.csv

Outputs:
    data/synthetic_clean.csv
    evaluation/figures/tsne.png        — t-SNE plot (real + synthetic, colored by label/source)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "evaluation", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

SEED = 42


def load_sentence_model():
    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return model


def encode_texts(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Cosine similarity = dot product after normalization
    )


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pair-wise cosine similarity. Embeddings must be L2-normalized."""
    return embeddings @ embeddings.T


def deduplicate_class(texts: list[str], embeddings: np.ndarray, threshold: float = 0.95):
    """
    Remove near-duplicates within a class using cosine similarity > threshold.
    Greedy: keep first occurrence, remove all subsequent near-duplicates.
    """
    n = len(texts)
    kept = []
    removed = 0
    mask = np.ones(n, dtype=bool)

    for i in tqdm(range(n), desc="Deduplicating", leave=False):
        if not mask[i]:
            continue
        kept.append(i)
        # Compute similarity of i with all j > i
        sims = embeddings[i] @ embeddings[i + 1:].T if i + 1 < n else np.array([])
        near_dup_offsets = np.where(sims > threshold)[0]
        for offset in near_dup_offsets:
            j = i + 1 + offset
            if mask[j]:
                mask[j] = False
                removed += 1

    return kept, removed


def diversity_score(embeddings: np.ndarray, sample_size: int = 1000) -> float:
    """Mean pairwise cosine distance on a random subset (avoid O(n^2) blow-up)."""
    if len(embeddings) > sample_size:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(embeddings), sample_size, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings

    sim_matrix = emb @ emb.T
    np.fill_diagonal(sim_matrix, np.nan)  # Exclude self-similarity
    mean_sim = np.nanmean(sim_matrix)
    return float(1.0 - mean_sim)  # cosine distance


def generate_tsne(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, model):
    """Generate t-SNE plot of real + synthetic samples colored by label and source."""
    from sklearn.manifold import TSNE

    # Subsample for t-SNE performance
    TSNE_MAX = 2000
    real_sub = real_df.sample(min(len(real_df), TSNE_MAX // 2), random_state=SEED)
    syn_sub = synthetic_df.sample(min(len(synthetic_df), TSNE_MAX // 2), random_state=SEED)

    all_df = pd.concat([real_sub, syn_sub], ignore_index=True)
    texts = all_df["text"].tolist()

    print(f"\nEncoding {len(texts)} samples for t-SNE...")
    embeddings = encode_texts(model, texts)

    print("Running t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)  # Remove n_iter
    coords = tsne.fit_transform(embeddings)

    all_df["x"] = coords[:, 0]
    all_df["y"] = coords[:, 1]

    # Generate combined marker information
    all_df["source"] = all_df.get("source", "unknown")
    all_df["label_str"] = all_df["label"].map({1: "hate", 0: "non-hate"})
    all_df["source_marker"] = all_df["source"].map({"real": "o", "synthetic": "s"})

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("colorblind", 2)
    colors = {1: palette[0], 0: palette[1]}
    markers = {"real": "o", "synthetic": "s"}

    for source, marker in markers.items():
        for label_val, label_str in [(1, "hate"), (0, "non-hate")]:
            subset = all_df[(all_df["source"] == source) & (all_df["label"] == label_val)]
            ax.scatter(
                subset["x"], subset["y"],
                c=[colors[label_val]],
                marker=marker,
                alpha=0.5,
                s=15,
                label=f"{source} / {label_str}",
            )

    ax.set_title("t-SNE: Real vs. Synthetic (by label and source)", fontsize=11)
    ax.set_xlabel("t-SNE dim 1", fontsize=9)
    ax.set_ylabel("t-SNE dim 2", fontsize=9)
    ax.legend(fontsize=8, markerscale=1.5, loc="best")

    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "tsne.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] t-SNE plot saved -> {save_path}")


def main():
    raw_path = os.path.join(DATA_DIR, "synthetic_raw.csv")
    clean_path = os.path.join(DATA_DIR, "synthetic_clean.csv")
    real_path = os.path.join(DATA_DIR, "real_samples.csv")

    if not os.path.exists(raw_path):
        print(f"ERROR: {raw_path} not found. Run synthetic_generator.py first.")
        sys.exit(1)

    print(f"Loading synthetic data from {raw_path}...")
    synthetic_df = pd.read_csv(raw_path)
    print(f"  Total raw samples: {len(synthetic_df)}")

    model = load_sentence_model()

    report_rows = []
    all_kept_indices = []

    for label in [1, 0]:
        label_name = "hate" if label == 1 else "non-hate"
        subset = synthetic_df[synthetic_df["label"] == label].reset_index(drop=True)
        original_count = len(subset)

        print(f"\nEncoding {label_name} samples ({original_count} texts)...")
        texts = subset["text"].tolist()
        embeddings = encode_texts(model, texts)

        print(f"Deduplicating {label_name}...")
        kept_indices, removed_count = deduplicate_class(texts, embeddings, threshold=0.95)
        kept_embeddings = embeddings[kept_indices]

        div_score = diversity_score(kept_embeddings)
        report_rows.append({
            "Class": label_name,
            "Original": original_count,
            "After Dedup": len(kept_indices),
            "Removed": removed_count,
            "Diversity Score": f"{div_score:.2f}",
        })
        all_kept_indices.append(subset.iloc[kept_indices].copy())

    clean_df = pd.concat(all_kept_indices, ignore_index=True)
    clean_df.to_csv(clean_path, index=False)

    # Print report
    print("\n" + "=" * 70)
    print(f"{'Class':<12} | {'Original':<10} | {'After Dedup':<12} | {'Removed':<8} | {'Diversity Score'}")
    print("-" * 70)
    for row in report_rows:
        print(
            f"{row['Class']:<12} | {row['Original']:<10} | {row['After Dedup']:<12} | {row['Removed']:<8} | {row['Diversity Score']}"
        )
    print("=" * 70)
    print(f"\n[OK] Cleaned data saved -> {clean_path} ({len(clean_df)} samples)")

    # t-SNE
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        generate_tsne(real_df, clean_df, model)
    else:
        print(f"[WARN]  Skipping t-SNE: {real_path} not found.")


if __name__ == "__main__":
    main()
