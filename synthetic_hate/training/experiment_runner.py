"""
Step 6 — Experiment Runner: fine-tune MiniLM on all 8 dataset variants.

Usage:
    python training/experiment_runner.py              # CPU
    python training/experiment_runner.py --colab      # Colab T4 (fp16=True)

Inputs:
    data/filtered/manifest.json
    data/real_test.csv

Outputs:
    results/results.json      — all 8 conditions × 4 metrics

Features:
    - Resume logic: skips already-logged variant_names in results.json
    - Prints live table after each run
    - Colab flag: fp16=True + /content/ path prefix
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
SEED = 42


def load_manifest(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        data = json.load(f)
    return data["variants"]


def load_results(results_path: str) -> list[dict]:
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def save_results(results: list[dict], results_path: str):
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def print_results_table(results: list[dict]):
    print("\n" + "=" * 100)
    print(f"{'Variant':<25} | {'N Train':>8} | {'F1-Mac':>7} | {'Prec':>7} | {'Recall':>7} | {'Acc':>7} | {'Time(s)':>8}")
    print("-" * 100)
    for r in results:
        print(
            f"{r['variant_name']:<25} | {r['n_train_samples']:>8} | "
            f"{r['f1_macro']:>7.4f} | {r['precision']:>7.4f} | "
            f"{r['recall']:>7.4f} | {r['accuracy']:>7.4f} | "
            f"{r['training_time_seconds']:>8.0f}"
        )
    print("=" * 100)


def make_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    from datasets import Dataset
    hf_dataset = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = hf_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")
    return tokenized


def run_experiment(
    variant: dict,
    test_df: pd.DataFrame,
    results_path: str,
    existing_results: list[dict],
    use_fp16: bool = False,
    colab_prefix: str = "",
) -> dict | None:
    """Fine-tune MiniLM on one variant, evaluate on test set, return metrics dict."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import evaluate as ev

    variant_name = variant["name"]
    variant_path = variant["path"]

    # Handle colab path prefix
    if colab_prefix:
        variant_path = variant_path.replace(BASE_DIR, colab_prefix)

    # Check if CSV exists (for real_only it's in data/, others in data/filtered/)
    if not os.path.exists(variant_path):
        print(f"  [WARN]  Skipping {variant_name}: file not found at {variant_path}")
        return None

    train_df = pd.read_csv(variant_path)
    train_df = train_df[["text", "label"]].dropna()
    train_df["label"] = train_df["label"].astype(int)
    n_train = len(train_df)

    print(f"\n{'─'*60}")
    print(f"[RUN] Training on: {variant_name} | N={n_train}")
    print(f"{'─'*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = make_hf_dataset(train_df, tokenizer)
    test_dataset = make_hf_dataset(test_df, tokenizer)

    # Output dir for this run (temp, not saved)
    output_dir = os.path.join(RESULTS_DIR, f"tmp_{variant_name}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        fp16=use_fp16,
        save_strategy="no",
        seed=SEED,
        report_to="none",
        logging_steps=50,
        disable_tqdm=False,
    )

    metric_acc = ev.load("accuracy")
    metric_f1 = ev.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Evaluate on test set with full metric suite
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = test_df["label"].values

    f1_mac = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)

    result = {
        "variant_name": variant_name,
        "threshold": variant.get("threshold"),
        "n_train_samples": n_train,
        "synthetic_pct": variant.get("synthetic_pct", 0),
        "f1_macro": round(f1_mac, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "training_time_seconds": round(elapsed, 1),
    }

    print(f"  [OK] F1-macro: {f1_mac:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Accuracy: {acc:.4f}")
    print(f"  [TIME]  Training time: {elapsed:.0f}s")

    # Append and save
    existing_results.append(result)
    save_results(existing_results, results_path)

    # Clean up temp dir
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", action="store_true", help="Enable fp16 + Colab path prefix")
    args = parser.parse_args()

    use_fp16 = args.colab
    colab_prefix = "/content" if args.colab else ""

    if args.colab:
        os.makedirs("/content/drive/MyDrive/synthetic_hate/results", exist_ok=True)
        results_path = "/content/drive/MyDrive/synthetic_hate/results/results.json"
    else:
        results_path = os.path.join(RESULTS_DIR, "results.json")

    manifest_path = os.path.join(DATA_DIR, "filtered", "manifest.json")
    test_path = os.path.join(DATA_DIR, "real_test.csv")

    if not os.path.exists(manifest_path):
        print(f"ERROR: {manifest_path} not found. Run quality_filter.py first.")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found. Run step1_data.py first.")
        sys.exit(1)

    variants = load_manifest(manifest_path)
    test_df = pd.read_csv(test_path)
    test_df["label"] = test_df["label"].astype(int)

    existing_results = load_results(results_path)
    completed_names = {r["variant_name"] for r in existing_results}

    print(f"Found {len(completed_names)} already-completed variants: {completed_names or '—'}")
    print(f"Will run {len(variants) - len(completed_names)} variants.")

    for variant in variants:
        vname = variant["name"]
        if vname in completed_names:
            print(f"  [SKIP]  Skipping {vname} (already in results.json)")
            continue

        run_experiment(
            variant=variant,
            test_df=test_df,
            results_path=results_path,
            existing_results=existing_results,
            use_fp16=use_fp16,
            colab_prefix=colab_prefix,
        )

        print_results_table(existing_results)

    print("\n[DONE] All experiments complete!")
    print_results_table(existing_results)
    print(f"\nResults saved -> {results_path}")


if __name__ == "__main__":
    main()
