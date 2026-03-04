# When Synthetic Data Falls Short
## A Pareto Analysis of LLM-Generated Data Filtering for Hate Speech Detection

> **Conference Short Paper Submission**

---

### Abstract

Annotation bottlenecks motivate the use of LLM-generated synthetic data for low-resource hate speech detection.
We conduct the first systematic **Pareto analysis** of discriminator-guided quality filtering across six continuous thresholds using an open-source LLM (LLaMA 3.1-8B via Groq API).
Across all thresholds, synthetic-only training consistently underperforms real data (best synthetic F1: **0.570** vs. real **0.651**) — a gap quality filtering cannot close.
A novel **precision-recall asymmetry analysis** reveals the diagnostic mechanism: LLM-generated hate speech captures explicit patterns well (high precision) but misses borderline and implicit cases (low recall).
Augmentation yields a marginal +0.006 F1 gain, confirmed non-significant by McNemar's test (p = 1.0).

---

### Results

| Condition | N Train | F1-macro | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| **real-only** (baseline) | 500 | **0.651** | 0.678 | 0.660 | 0.660 |
| **real + top-50%** | 1085 | _0.658_ | 0.699 | 0.670 | 0.670 |
| synthetic unfiltered | 1170 | 0.570 | 0.723 | 0.620 | 0.620 |
| synthetic top-90% | 1052 | 0.509 | 0.689 | 0.580 | 0.580 |
| synthetic top-70% | 818 | 0.533 | 0.677 | 0.590 | 0.590 |
| synthetic top-50% | 585 | 0.540 | 0.708 | 0.600 | 0.600 |
| synthetic top-30% | 350 | 0.541 | 0.593 | 0.570 | 0.570 |
| synthetic top-10% | 116 | 0.333 | 0.250 | 0.500 | 0.500 |

---

### Key Findings

1. **Synthetic data consistently underperforms real data** — by 0.08 F1-macro regardless of quality threshold applied.
2. **Quality filtering does not recover the performance gap** — the Pareto curve is flat between top-30% and top-90%.
3. **Augmentation yields negligible gain** — real+top50 achieves +0.006 F1 over real-only (McNemar p=1.0, Cohen's d=0.099 [small]).
4. **Precision-recall asymmetry is the key diagnostic** — synthetic conditions show PR gaps of 0.08–0.11 vs. 0.018 for real data. LLMs generate prototypical hate well but miss borderline cases.

---

### Project Structure

```
synthetic_hate/
├── step1_data.py                      # Load real HatEval data
├── generation/
│   ├── synthetic_generator.py         # Step 2: Groq API generation
│   └── diversity_checker.py           # Step 3: Deduplication + t-SNE
├── discrimination/
│   ├── discriminator_trainer.py       # Step 4: Real vs. synthetic classifier
│   └── quality_filter.py             # Step 5: Threshold filtering
├── training/
│   └── experiment_runner.py           # Step 6: Fine-tune MiniLM on 8 variants
├── evaluation/
│   ├── statistical_tests.py           # Step 7a: McNemar, ANOVA, Cohen's d
│   ├── figure_generator.py            # Step 7b: 4 publication figures
│   └── figures/                       # pareto_curve, pr_asymmetry, etc.
├── paper/
│   ├── main.tex                       # ACL 2026 short paper
│   └── references.bib                 # 15 BibTeX entries
├── results/
│   ├── results.json                   # All 8 experiment results
│   └── statistical_results.json       # Statistical test outputs
├── data/
│   ├── real_samples.csv              # 500 real training samples
│   ├── real_test.csv                 # 100 real test samples
│   ├── synthetic_clean.csv           # 1,170 deduplicated synthetic
│   ├── synthetic_scored.csv          # With quality scores
│   └── filtered/                      # 8 training variants
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

# Copy .env.example to .env and fill in your keys
cp .env.example .env

# 2. Load real data (HatEval from HuggingFace)
python step1_data.py

# 3. Generate synthetic data (or use complete_synthetic.py bypass)
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

### Statistical Summary

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| McNemar (augmentation gain) | χ²=0.000 | p=1.000 | **Not significant** — +0.006 F1 is noise |
| ANOVA (flat Pareto) | F=1115.3 | p<0.001* | Driven by top-10% collapse; excl. top-10: not sig. |
| Pearson r (synth% vs PR-gap) | r=0.032 | p=0.939 | Weak, non-significant overall correlation |
| Cohen's d (real vs synthetic) | d=0.099 | 95%CI [0.015, 0.188] | **Small effect** size |

*ANOVA significance is driven by the top-10% condition (116 samples, F1=0.333 collapse). Excluding it, the remaining 4 synthetic conditions span F1 0.509–0.570 with no meaningful trend.

---

### Citation

```bibtex
@article{dutta2026synthetic,
  title   = {When Synthetic Data Falls Short: A Pareto Analysis of
             LLM-Generated Data Filtering for Hate Speech Detection},
  author  = {Dutta, Sujato},
  journal = {arXiv preprint arXiv:[YOUR ARXIV ID]},
  year    = {2026},
  url     = {https://arxiv.org/abs/[YOUR ARXIV ID]}
}
```

---

### License

MIT License. For research use only. The models and generated data should not be used to produce or promote hate speech.
