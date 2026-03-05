"""
Step 1 -- Load real HatEval dataset and split into train/test pools.

Dataset: Lots-of-LoRAs/task333_hateeval_classification_hate_en
(Fully public, ~6k rows, SemEval-2019 HatEval binary classification)

Columns in raw dataset:
  input  : long NLP-task prompt ending with "Post: <tweet>"
  output : numpy array like array(['Non-hateful'], dtype=object) or array(['hateful'])

Usage:    python step1_data.py

Outputs:
  data/real_samples.csv  -- 500 samples (250 hate + 250 non-hate)
  data/real_test.csv     -- 400 samples (200 hate + 200 non-hate), never touched in training
"""

import os, re, sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

SEED     = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

load_dotenv(os.path.join(BASE_DIR, ".env"))
DATASET_ID = "Lots-of-LoRAs/task333_hateeval_classification_hate_en"


def extract_tweet(raw: str) -> str:
    """
    The prompt format is:
      Definition: ...\n\nPositive Example 1 - ...\n\nNow complete ...\n\nInput: Post: <tweet>\nOutput:
    We extract the LAST occurrence of 'Post: ' to get the actual tweet.
    """
    raw = str(raw)
    # Find last "Post: " occurrence
    idx = raw.rfind("Post: ")
    if idx != -1:
        after = raw[idx + len("Post: "):].strip()
        # Remove trailing "\nOutput:" if present
        after = re.split(r"\n\s*Output\s*:", after)[0].strip()
        return after
    # Fallback: last line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1] if lines else raw.strip()


def parse_label(val) -> int:
    """Convert 'hateful'/'Non-hateful' (possibly inside numpy array) to 1/0."""
    if isinstance(val, (list, np.ndarray)):
        val = val[0] if len(val) > 0 else ""
    s = str(val).strip().lower()
    if s in ("hateful", "yes", "1", "hate", "true"):
        return 1
    if s in ("non-hateful", "non hateful", "no", "0", "not hate", "false"):
        return 0
    return -1


def load_hateval() -> pd.DataFrame:
    print("Loading:", DATASET_ID)
    # Repackaged HatEval (Basile et al., 2019) — original: valeriobasile/HatEval
    ds = load_dataset(DATASET_ID)

    frames = []
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        print("  split '{}': {} rows".format(split_name, len(df)))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    combined["text"]  = combined["input"].apply(extract_tweet)
    combined["label"] = combined["output"].apply(parse_label)

    # Keep valid labels only
    combined = combined[combined["label"].isin([0, 1])][["text", "label"]].dropna()
    combined = combined[combined["text"].str.len() > 5]
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Clean samples:", len(combined))
    print("  hate   (1):", (combined.label == 1).sum())
    print("  non-hate(0):", (combined.label == 0).sum())
    return combined


def sample_balanced(df, n_per_class, exclude_indices=None):
    if exclude_indices is None:
        exclude_indices = set()
    avail = df[~df.index.isin(exclude_indices)]
    h  = avail[avail.label == 1].sample(n=n_per_class, random_state=SEED)
    nh = avail[avail.label == 0].sample(n=n_per_class, random_state=SEED)
    return pd.concat([h, nh]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    df = load_hateval()

    n_hate    = (df.label == 1).sum()
    n_nonhate = (df.label == 0).sum()
    assert n_hate    >= 450, "Need >=450 hate for train+test, got {}".format(n_hate)
    assert n_nonhate >= 450, "Need >=450 non-hate for train+test, got {}".format(n_nonhate)

    # TEST set first (200+200=400) -- held out permanently, zero overlap with train
    test_df = sample_balanced(df, n_per_class=200)
    test_df["source"] = "real"
    test_idx = set(test_df.index)

    # TRAIN pool from remainder (250+250=500)
    train_df = sample_balanced(df, n_per_class=250, exclude_indices=test_idx)
    train_df["source"] = "real"

    train_path = os.path.join(DATA_DIR, "real_samples.csv")
    test_path  = os.path.join(DATA_DIR, "real_test.csv")
    train_df[["text","label","source"]].to_csv(train_path, index=False)
    test_df [["text","label","source"]].to_csv(test_path,  index=False)

    print("\n" + "="*55)
    print("[DONE] Training pool ->", train_path)
    print("       hate:", (train_df.label==1).sum(),
          "| non-hate:", (train_df.label==0).sum(),
          "| total:", len(train_df))
    print("[DONE] Test set      ->", test_path)
    print("       hate:", (test_df.label==1).sum(),
          "| non-hate:", (test_df.label==0).sum(),
          "| total:", len(test_df))
    print("="*55)
    print("\n--- Sample rows (training) ---")
    for _, row in train_df[["text","label"]].head(3).iterrows():
        print("  [{}] {}".format(int(row.label), row.text.encode("ascii","replace").decode()[:120]))
    print("\n--- Sample rows (test) ---")
    for _, row in test_df[["text","label"]].head(3).iterrows():
        print("  [{}] {}".format(int(row.label), row.text.encode("ascii","replace").decode()[:120]))


if __name__ == "__main__":
    main()
