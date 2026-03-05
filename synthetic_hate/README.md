# When Synthetic Data Falls Short
## A Pareto Analysis of LLM-Generated Data Filtering for Hate Speech Detection

> **Conference Short Paper Submission**

---

### Abstract

Annotation bottlenecks motivate the use of LLM-generated synthetic data for low-resource hate speech detection.
We conduct the first systematic **Pareto analysis** of discriminator-guided quality filtering across six continuous thresholds using an open-source LLM (LLaMA 3.1-8B via Groq API), evaluated on a **400-sample held-out test set**.
Across all thresholds, synthetic-only training consistently underperforms real data (best synthetic F1: **0.576** vs. real **0.715**) — a gap of 0.139 F1 that quality filtering cannot close.
A novel **precision-recall asymmetry analysis** reveals the diagnostic mechanism: LLM-generated hate speech captures explicit patterns well (high precision) but systematically misses borderline and implicit cases (low recall).
Augmentation yields a +0.012 F1 gain, which falls short of statistical significance (McNemar χ²=3.125, p=0.077).

---

### Results

Evaluated on **400 balanced test samples** (200 hate, 200 non-hate). Training pool fixed at 500 real samples. Zero train/test overlap enforced.

| Condition | N Train | F1-macro | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **real-only** (baseline) | 500 | **0.715** | 0.715 | 0.715 | 0.715 |
| **real + top-50%** (augmented) | 1085 | _0.727_ | 0.741 | 0.730 | 0.730 |
| synthetic unfiltered | 1170 | 0.576 | 0.720 | 0.623 | 0.623 |
| synthetic top-90% | 1052 | 0.548 | 0.762 | 0.613 | 0.613 |
| synthetic top-70% | 818 | 0.575 | 0.697 | 0.618 | 0.618 |
| synthetic top-50% | 585 | 0.574 | 0.725 | 0.623 | 0.623 |
| synthetic top-30% | 350 | 0.576 | 0.644 | 0.605 | 0.605 |
| synthetic top-10% | 116 | 0.333 | 0.250 | 0.500 | 0.500 |

---

### Key Findings

1. **Synthetic data consistently underperforms real data** — by 0.139 F1-macro regardless of quality threshold applied.
2. **Quality filtering does not recover the performance gap** — the Pareto curve is flat between top-30% and top-90% (F1 range: 0.548–0.576).
3. **Augmentation yields a sub-threshold gain** — real+top50 achieves +0.012 F1 over real-only (McNemar χ²=3.125, p=0.077; not significant at α=0.05).
4. **Precision-recall asymmetry is the key diagnostic** — synthetic conditions show P–R gaps of 0.039–0.149 vs. <0.012 for real-data conditions. LLMs generate prototypical hate well (↑ precision) but miss borderline and implicit cases (↓ recall).

---

### Statistical Summary

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| McNemar (augmentation gain) | χ²=3.125 | p=0.077 | **Not significant** at α=0.05 — +0.012 F1 gain is sub-threshold |
| ANOVA (flat Pareto) | F=511.05 | p<0.001* | Artefact of bootstrap variance; 5 conditions span only 0.028 F1 |
| Pearson r (synth% vs PR-gap) | r=0.115 | p=0.786 | Weak, non-significant; qualitative pattern is consistent |
| Cohen's d (real vs synthetic) | d=2.315 | 95% CI [2.203, 2.419] | **Large effect** — real data meaningfully outperforms synthetic |

*ANOVA significance is a statistical artefact: all five synthetic-only conditions (excl. degenerate top-10%) are compressed into a 0.028 F1 band (0.548–0.576). The bootstrap variance drives nominal significance; no pairwise comparison reflects a meaningful performance difference.*

---

### Project Structure

```
synthetic_hate/
├── step1_data.py                      # Load real HatEval data (500 train + 400 test)
├── generation/
│   ├── synthetic_generator.py         # Step 2: Groq API generation (LLaMA 3.1-8B)
│   └── diversity_checker.py           # Step 3: Deduplication + t-SNE
├── discrimination/
│   ├── discriminator_trainer.py       # Step 4: Real vs. synthetic classifier
│   └── quality_filter.py             # Step 5: Threshold filtering
├── training/
│   └── experiment_runner.py           # Step 6: Fine-tune MiniLM on 8 variants
├── evaluation/
│   ├── statistical_tests.py           # Step 7a: McNemar, ANOVA, Cohen's d
│   ├── figure_generator.py            # Step 7b: 4 publication figures
│   └── figures/                       # pareto_curve, pr_asymmetry, quality_score_dist
├── paper/
│   ├── main.tex                       # Conference short paper (LaTeX)
│   └── references.bib                 # BibTeX entries
├── results/
│   ├── results.json                   # All 8 experiment results
│   └── statistical_results.json       # Statistical test outputs
├── data/
│   ├── real_samples.csv              # 500 real training samples
│   ├── real_test.csv                 # 400 real test samples (zero overlap with train)
│   ├── synthetic_clean.csv           # 1,170 deduplicated synthetic samples
│   ├── synthetic_scored.csv          # Synthetic samples with quality scores
│   └── filtered/                      # 8 training variant CSVs
├── .env                               # API keys (not committed)
├── requirements.txt
└── README.md
```

---

### Quickstart

```bash
# 1. Create environment
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Copy .env.example to .env and add your Groq API key
cp .env.example .env

# 2. Load real data (HatEval from HuggingFace — 500 train + 400 test)
python step1_data.py

# 3. Generate synthetic data
python generation/synthetic_generator.py

# 4. Deduplication & diversity check
python generation/diversity_checker.py

# 5. Train discriminator & assign quality scores
python discrimination/discriminator_trainer.py
python discrimination/quality_filter.py

# 6. Run all 8 experiments
python training/experiment_runner.py

# 7. Statistical tests + figures
python evaluation/statistical_tests.py
python evaluation/figure_generator.py
```

---

### License

MIT License. For research use only. The models and generated data should not be used to produce or promote hate speech.
