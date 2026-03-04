"""
Step 2 — Synthetic Data Generator using Groq API.

Usage:
    python generation/synthetic_generator.py --n_samples 5000

Outputs:
    data/synthetic_raw.csv — raw Groq-generated samples (incremental, crash-safe)

Features:
    - 5 prompt templates per class across diverse axes
    - Incremental saves every 50 samples
    - Resume logic: reads existing CSV and continues from offset
    - 2s sleep between API calls (Groq free tier: 30 req/min)
    - 429 backoff: sleep 60s, retry once
    - Malformed JSON: skip batch + log failure
"""

import os
import sys
import json
import time
import argparse
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env from parent directory
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

try:
    from groq import Groq  # type: ignore
except ImportError:
    print("ERROR: groq package not installed. Run: pip install groq")
    sys.exit(1)

SEED = 42
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "synthetic_raw.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# ─── Prompt Templates ────────────────────────────────────────────────────────

HATE_TEMPLATES = [
    # T1 — Direct explicit language targeting gender in professional contexts
    (1, """Generate 5 hate speech examples that use direct, explicit language targeting women in professional or workplace contexts. 
These should include derogatory statements about women's competence or right to hold professional roles.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T2 — Coded/implicit language with plausible deniability
    (2, """Generate 5 hate speech examples that use coded or implicit language targeting women, maintaining plausible deniability.
These should be subtle — using dog whistles, euphemisms, or framing that implies misogyny without stating it outright.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T3 — Dehumanization framing
    (3, """Generate 5 hate speech examples that use dehumanization framing toward women — comparing them to animals, objects, or inferior categories.
These examples should strip agency or humanity from women as a group.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T4 — Role exclusion ("women don't belong in X")
    (4, """Generate 5 hate speech examples arguing women do not belong in specific fields, roles, or spaces (e.g., tech, politics, leadership, military).
These should assert explicit exclusion based on gender.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T5 — Condescending/patronizing framing
    (5, """Generate 5 hate speech examples that use condescending or patronizing language toward women — treating them as intellectually inferior, overly emotional, or in need of male guidance.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),
]

NON_HATE_TEMPLATES = [
    # T1 — Neutral professional statement
    (1, """Generate 5 neutral, professional statements about gender diversity in the workplace. These should be completely factual and non-judgmental.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T2 — Counter-speech directly challenging hate
    (2, """Generate 5 counter-speech examples that directly challenge or refute misogynistic arguments without themselves being hateful.
These should be assertive rebuttals to sexist claims.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T3 — Borderline ambiguous (critical but not hateful)
    (3, """Generate 5 examples that critique gender dynamics or raise critical questions about gender roles in a way that is blunt or controversial but NOT hateful.
These should be borderline — edgy but ultimately non-hateful.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T4 — Supportive/empowering language
    (4, """Generate 5 supportive or empowering statements about women's rights, capabilities, or achievements. These should be positive and affirming.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),

    # T5 — Factual/informational statement
    (5, """Generate 5 factual, informational statements about gender statistics, research findings, or policy discussions. Tone should be academic and neutral.
Return ONLY a valid JSON array of 5 objects, each with key "text" (string). No explanations, no markdown, no preamble.
Example format: [{"text": "..."}, {"text": "..."}, ...]"""),
]

INTENSITIES = ["mild", "moderate", "strong"]


def call_groq_with_retry(client, prompt: str, model: str = "llama-3.1-8b-instant") -> str | None:
    """Call Groq API with one retry on 429 rate-limit errors."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                print(f"\n[WARN]  Rate limit hit. Sleeping 60s before retry...")
                time.sleep(60)
            else:
                print(f"\n[WARN]  API error: {e}")
                return None
    return None


def parse_json_array(raw_text: str) -> list[dict] | None:
    """Extract and parse a JSON array from LLM output."""
    if raw_text is None:
        return None
    # Try to find JSON array inside the response
    try:
        start = raw_text.index("[")
        end = raw_text.rindex("]") + 1
        json_str = raw_text[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return None


def load_existing(output_path: str) -> pd.DataFrame:
    """Load existing synthetic_raw.csv for resume logic."""
    if os.path.exists(output_path):
        df = pd.read_csv(output_path)
        print(f"[FOLDER] Resuming: found {len(df)} existing samples in {output_path}")
        return df
    return pd.DataFrame()


def generate_samples(n_samples: int):
    """Main generation loop."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("ERROR: GROQ_API_KEY not set in .env file.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    existing_df = load_existing(OUTPUT_PATH)
    existing_samples = existing_df.to_dict("records") if len(existing_df) > 0 else []

    existing_hate = sum(1 for s in existing_samples if s.get("label") == 1)
    existing_non_hate = sum(1 for s in existing_samples if s.get("label") == 0)

    target_per_class = n_samples // 2  # 2500 each
    need_hate = max(0, target_per_class - existing_hate)
    need_non_hate = max(0, target_per_class - existing_non_hate)

    print(f"\n[TARGET] Target: {n_samples} total ({target_per_class} per class)")
    print(f"   Need: {need_hate} more hate | {need_non_hate} more non-hate")

    new_samples = []
    failed_batches = 0
    save_counter = 0

    def save_progress():
        all_samples = existing_samples + new_samples
        df = pd.DataFrame(all_samples)
        df.to_csv(OUTPUT_PATH, index=False)

    # Build task queue: (label, templates_list, remaining_count)
    tasks = []
    if need_hate > 0:
        tasks.append((1, HATE_TEMPLATES, need_hate))
    if need_non_hate > 0:
        tasks.append((0, NON_HATE_TEMPLATES, need_non_hate))

    for label, templates, needed in tasks:
        label_name = "hate" if label == 1 else "non-hate"
        generated = 0
        template_cycle = 0

        pbar = tqdm(total=needed, desc=f"Generating {label_name}", unit="samples")

        while generated < needed:
            template_id, prompt = templates[template_cycle % len(templates)]
            template_cycle += 1

            raw = call_groq_with_retry(client, prompt)
            time.sleep(2)  # Groq free tier rate limit

            parsed = parse_json_array(raw)
            if parsed is None:
                failed_batches += 1
                print(f"\n[WARN]  Failed batch #{failed_batches} (template {template_id}, {label_name}) — skipping")
                continue

            for item in parsed:
                if not isinstance(item, dict) or "text" not in item:
                    continue
                text = str(item.get("text", "")).strip()
                if len(text) < 10:
                    continue

                intensity = random.choice(INTENSITIES)
                new_samples.append({
                    "text": text,
                    "label": label,
                    "template_id": template_id,
                    "target_group": "gender",
                    "intensity": intensity,
                    "is_synthetic": True,
                    "source": "synthetic",
                })
                generated += 1
                save_counter += 1
                pbar.update(1)

                # Incremental save every 50 samples
                if save_counter % 50 == 0:
                    save_progress()
                    h = existing_hate + sum(1 for s in new_samples if s["label"] == 1)
                    nh = existing_non_hate + sum(1 for s in new_samples if s["label"] == 0)
                    total = h + nh
                    pbar.set_postfix(
                        total=total,
                        hate=h,
                        non_hate=nh,
                        failed=failed_batches,
                    )

                if generated >= needed:
                    break

        pbar.close()

    # Final save
    save_progress()
    total_saved = len(existing_samples) + len(new_samples)

    print(f"\n[OK] Generation complete!")
    print(f"   Total samples: {total_saved}")
    final_hate = sum(1 for s in (existing_samples + new_samples) if s["label"] == 1)
    final_nh = total_saved - final_hate
    print(f"   hate: {final_hate} | non-hate: {final_nh}")
    print(f"   Failed batches: {failed_batches}")
    print(f"   Saved to: {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hate speech data via Groq API")
    parser.add_argument("--n_samples", type=int, default=5000, help="Total samples to generate")
    args = parser.parse_args()

    generate_samples(args.n_samples)


if __name__ == "__main__":
    main()
